from __future__ import print_function, absolute_import

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import math
from tqdm import tqdm_notebook as tqdm

@cuda.jit
def move(rng_states, rd, ro, oc, pOut, doms, rs, domhits, domhitstimes, pa, ps, Nphotons):

    thread_id = cuda.grid(1)
    if thread_id < Nphotons:

        def rng():
            return xoroshiro128p_uniform_float32(rng_states, thread_id)

        def ln(p):
            ranNum = rng()
            return -math.log(ranNum)*p

        def domhit(center, radius, ro, rd, oc, time, t_min, t_max):
            oc[0] = ro[0] - center[0]
            oc[1] = ro[1] - center[1]
            a = 0
            a = rd.dot(rd)
            b = 0
            b = oc.dot(rd)
            c = 0
            c = oc.dot(oc) 
            c -= radius*radius
            discriminant = b*b - a*c
            if discriminant > 0:
                temp = (-b - math.sqrt(b*b - a*c))/a
                if (temp < t_max) and (temp > t_min):
                    ro[0] += rd[0]*temp
                    ro[1] += rd[1]*temp
                    time = temp
                    return True 

                temp = (-b + math.sqrt(b*b - a*c))/a
                if temp < t_max and temp > t_min:
                    ro[0] += rd[0]*temp
                    ro[1] += rd[1]*temp
                    time = temp
                    return True 
            return False 

        def get_random_dir(rd):
            d = rng()*math.pi*2
            vx = math.cos(d)
            vy = math.sin(d)
            rd[0] = vx
            rd[1] = vy


        thread_id = 0
        if thread_id < Nphotons:
            get_random_dir(rd) 
            pDist = ln(pa)
            scatter = ln(ps)
            t = 0
            tHit = 0
            while scatter < pDist:

                for i in range(len(doms)):
                    center = doms[i]
                    radius = rs[i]
                    if domhit(center, radius, ro, rd, oc, tHit, 0, scatter):
                        domhits[thread_id, i] += 1
                        domhitstimes[thread_id, i] = t + tHit
                        pOut[thread_id, 0] = ro[0]
                        pOut[thread_id, 1] = ro[1]
                        return

                t += scatter
                ro[0] = rd[0]*scatter
                ro[1] = rd[1]*scatter
                pDist = pDist - scatter
                get_random_dir(rd)
                scatter = ln(ps)
                
            for i in range(len(doms)):
                center = doms[i]
                radius = rs[i]
                if domhit(center, radius, ro, rd, oc, tHit, 0, pDist):
                    domhits[thread_id, i] += 1
                    domhitstimes[thread_id, i] = t + tHit
                    pOut[thread_id, 0] = ro[0]
                    pOut[thread_id, 1] = ro[1]
                    return

            pOut[thread_id, 0] = np.nan
            pOut[thread_id, 1] = np.nan


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

def simulate(Nexp, N, initial_poisition, doms, rs, threads_per_block, pa=0.01, ps=0.02, seed=None, verbose=True):
    if verbose:
        print ( Nexp )
    import time
    blocks = N//threads_per_block + 1
    # N = threads_per_block * blocks
    ro = np.array(initial_poisition, dtype=np.float32)
    rd = np.zeros((1,2), dtype=np.float32)
    oc = np.zeros((1,2), dtype=np.float32)

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
        pOut = np.zeros((N,2), dtype=np.float32)
        # Create empty array for domhits
        domhits = np.zeros((N, len(rs)), dtype=np.int32)
        domhitstimes = np.zeros((N, len(rs)), dtype=np.int32)

        # Calculate x, y and domhits
        move[blocks, threads_per_block](rng_states, rd, ro, oc, pOut, 
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