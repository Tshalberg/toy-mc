from __future__ import print_function, absolute_import

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import math
from tqdm import tqdm_notebook as tqdm

@cuda.jit
def move(rng_states, start_x, start_y, out_x, out_y, doms, rs, domhits, domhitstimes, pa, ps, Nphotons):
    """Find the maximum value in values and store in result[0]"""
    thread_id = cuda.grid(1)
    
    if thread_id < Nphotons:
    # if thread_id < 200:

        def rng():
            return xoroshiro128p_uniform_float32(rng_states, thread_id)
        
        x = start_x
        y = start_y
        d = rng()*math.pi*2
        vx = math.cos(d)
        vy = math.sin(d)
        absorbed = False
        time = 0
        while not absorbed:
            if rng() < ps:#1:
                d = xoroshiro128p_uniform_float32(rng_states, thread_id)*math.pi*2
                vx = math.cos(d)
                vy = math.sin(d)
            if rng() < pa:#05:
                absorbed = True
            x += vx
            y += vy
            for i in range(len(doms)):
                domx = doms[i,0]
                domy = doms[i,1]
                r = rs[i]
                if r >= (math.sqrt((domx-x)**2 + (domy-y)**2)):
                    domhits[thread_id, i] += 1
                    domhitstimes[thread_id, i] = time
                    absorbed = True
            time += 1

        out_x[thread_id] = x
        out_y[thread_id] = y


def create_doms(xlim, ylim, dx, dy, r):
    xs = np.arange(xlim[0], xlim[1]+0.01, dx)
    ys = np.arange(ylim[0], ylim[1]+0.01, dy)

    centers = []
    for x in xs:
        for y in ys:
            if x != 0 or y != 0:
                centers.append([x, y])

    centers = np.array(centers)
    radii = np.ones(len(centers))*r

    return [centers, radii]

def simulate(Nexp, N, initial_position, doms, rs, threads_per_block, pa=0.01, ps=0.02, seed=None, verbose=True):
    if verbose:
        print ( Nexp )
    import time
    blocks = N//threads_per_block + 1
    # N = threads_per_block * blocks
    x_start, y_start = np.array(initial_position, dtype=np.float32)

    domhits_all = []
    domhitstimes_all = []
    x_y = []
    t1 = time.time()
    for i in range(Nexp):
    # for i in tqdm(range(Nexp)):
        if seed is None:
            # Set a random seed
            RanSeed = np.random.randint(1, 123456)

        # Initialize the random states for the kernel
        rng_states = create_xoroshiro128p_states(N, seed=RanSeed)
        # Create empty arrays for the (x, y) values
        out_x, out_y = np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)
        # Create empty array for domhits
        domhits = np.zeros((N, len(rs)), dtype=np.int32)
        domhitstimes = np.zeros((N, len(rs)), dtype=np.int32)

        # Calculate x, y and domhits
        move[blocks, threads_per_block](rng_states, x_start, y_start, out_x, out_y, 
                                        doms, rs, domhits, domhitstimes, pa, ps, N)
        # Save the hit information
        domhits = np.sum(domhits, axis=0)
        domhits_all.append(domhits)
        domhitstimes_all.append(domhitstimes)
        x_y.append([x_start, y_start])
    
    t2 = time.time()
    if verbose:
        print (t2-t1)
    x_y = np.array(x_y)
    domhits_all = np.array(domhits_all)
    return domhits_all, domhitstimes_all