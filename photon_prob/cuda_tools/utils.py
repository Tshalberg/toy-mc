import numpy as np
import pycuda.tools
from pycuda import characterize
import pycuda.driver as cuda
import pycuda.compiler
import pycuda.autoinit
from pycuda import gpuarray as ga
import time
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

def get_rng_states(module, size, seed=1):
    "Return `size` number of CUDA random number generator states."
    rng_states = cuda.mem_alloc(size*characterize.sizeof('curandStatePhilox4_32_10_t', '#include <curand_kernel.h>'))
    # print "rng_states: ", size*characterize.sizeof('curandStatePhilox4_32_10_t', '#include <curand_kernel.h>')
    
    # module = pycuda.compiler.SourceModule(kernel_code, no_extern_c=True, 
    #    include_dirs=['/home/thomas/Documents/toy-mc/photon_prob/cuda_tools'])

    init_rng = module.get_function('init_rng')

    init_rng(np.uint64(size), rng_states, np.uint64(seed), block=(64,1,1), grid=(size//64+1,1))

    return rng_states



def get_times(module, size):
    "Return `size` number of CUDA random number generator states."
    times = cuda.mem_alloc(size*characterize.sizeof('float'))
    # print "times: ", size*characterize.sizeof('float')

    # module = pycuda.compiler.SourceModule(kernel_code, no_extern_c=True, 
    #    include_dirs=['/home/thomas/Documents/toy-mc/photon_prob/cuda_tools'])
    
    init_times = module.get_function('init_times')

    init_times(np.uint64(size), times, block=(64,1,1), grid=(size//64+1,1))

    return times


def get_doms(module, size, radius, d, dN):
    "Return `size` number of CUDA random number generator states."
    d_list = cuda.mem_alloc(size*characterize.sizeof('dom', "#include <dom_RT.h>", include_dirs='/home/thomas/Documents/toy-mc/photon_prob/cuda_tools'))
   # hits = cuda.mem_alloc(size*characterize.sizeof('int'))
    # print "doms: ", size*characterize.sizeof('dom', "#include <dom_RT.h>", include_dirs='/home/thomas/Documents/toy-mc/photon_prob/cuda_tools')

    # module = pycuda.compiler.SourceModule(kernel_code, no_extern_c=True, 
    #    include_dirs=['/home/thomas/Documents/toy-mc/photon_prob/cuda_tools'])
    
    create_doms = module.get_function('create_doms')

    d = np.uint32(d)
    dN = np.uint32(dN)
    radius = np.float32(radius)
    create_doms(radius, d, dN, d_list, block=(64,1,1), grid=(size//64+1,1))

    return d_list #, hits

def get_pInit(module, x, y, z):
    "Return `size` number of CUDA random number generator states."
    pInit = cuda.mem_alloc(characterize.sizeof('vec3', "#include <vec3.h>", include_dirs='/home/thomas/Documents/toy-mc/photon_prob/cuda_tools'))

    # module = pycuda.compiler.SourceModule(kernel_code, no_extern_c=True, 
    #    include_dirs=['/home/thomas/Documents/toy-mc/photon_prob/cuda_tools'])
    
    init_pInit = module.get_function('init_pInit')

    init_pInit(np.uint64(1), np.float32(x), np.float32(y), np.float32(z), pInit, block=(1,1,1), grid=(1,1))

    return pInit
    

def simulate_photons(module, Nobs, Nruns, oversampling, datahits, datatimes, radius, d, dN, x, y, z, pa, ps, seed=666, Nthreads=64):
    
    Nphotons = Nobs*Nruns*oversampling
    Nobs *= oversampling
    print "Total Threads: %s" % Nphotons
    assert(Nphotons <= 1.1e8)

    d = np.uint32(d)
    dN = np.uint32(dN)
    radius = np.float32(radius)
    Ndoms = np.uint32(pow((d/dN)*2+1, 3) - 1)

    t1 = time.time()
    rng_states = get_rng_states(module, Nphotons, seed=seed)
    t2 = time.time()

    d_list = get_doms(module, Ndoms, radius, d, dN)
    t3 = time.time()
    pInit = get_pInit(module, x, y, z)
    t4 = time.time()
    
    print "t2-t1: ", t2-t1
    print "t3-t2: ", t3-t2
    print "t4-t3: ", t4-t3

    # for i in range(Nruns):
    start = time.time()
    datahits = np.zeros(Ndoms*Nruns, dtype=np.int32)
    datatimesbinned = np.zeros(Ndoms*Nruns, dtype=np.float32)
    datatimes = np.zeros(Nphotons, dtype=np.float32)
    datapositions = np.zeros(Nphotons*3, dtype=np.float32)
    simulate = module.get_function('simulate')

    simulate(np.uint64(Nphotons), np.uint64(Nobs), rng_states, d_list, cuda.InOut(datahits), cuda.InOut(datatimes), cuda.InOut(datatimesbinned), 
             cuda.InOut(datapositions), pInit, np.float32(pa), np.float32(ps),np.uint32(Ndoms), 
             block=(Nthreads, 1, 1), grid=(Nphotons//Nthreads + 1, 1))    

    print "end-start", time.time() - start
    print "sumHits: ", sum(datahits)
    print 

    datahits = np.reshape(np.array(datahits, dtype=float), (Nruns, Ndoms))/oversampling
    datatimesbinned = np.reshape(np.array(datatimesbinned, dtype=float), (Nruns, Ndoms))


    return datahits, np.array(datatimes, dtype=float), datatimesbinned, datapositions

def simulate_grid(module, Nobs, N, oversampling, datahits, datatimes, radius, d, dN, pa, ps, seed=666, Nthreads=64):
    
    Nruns = N*N
    Nphotons = Nobs*Nruns*oversampling
    Nobs *= oversampling
    print "Total Threads: %s" % Nphotons
    assert(Nphotons <= 1.1e8)

    d = np.uint32(d)
    dN = np.uint32(dN)
    radius = np.float32(radius)
    Ndoms = np.uint32(pow((d/dN)*2+1, 3) - 1)

    t1 = time.time()
    rng_states = get_rng_states(module, Nphotons, seed=seed)
    t2 = time.time()

    d_list = get_doms(module, Ndoms, radius, d, dN)
    t3 = time.time()
    x = np.linspace(-20, 20, N)
    y = np.linspace(-20, 20, N)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(N*N)
    pInit = np.concatenate([X.flatten(), Y.flatten(), Z]).astype(np.float32)
    t4 = time.time()
    
    print "t2-t1: ", t2-t1
    print "t3-t2: ", t3-t2
    print "t4-t3: ", t4-t3

    start = time.time()
    datahits = np.zeros(Ndoms*Nruns, dtype=np.int32)
    datatimesbinned = np.zeros(Ndoms*Nruns, dtype=np.float32)
    datatimes = np.zeros(Nphotons, dtype=np.float32)
    datapositions = np.zeros(Nphotons*3, dtype=np.float32)
    simulate = module.get_function('simulate_grid')

    simulate(np.uint64(Nphotons), np.uint64(Nobs), rng_states, d_list, cuda.InOut(datahits), cuda.InOut(datatimes), cuda.InOut(datatimesbinned), 
             cuda.InOut(datapositions), cuda.In(pInit), np.float32(pa), np.float32(ps),np.uint32(Ndoms), 
             block=(Nthreads, 1, 1), grid=(Nphotons//Nthreads + 1, 1))    

    print "end-start", time.time() - start
    print "sumHits: ", sum(datahits)
    print 

    datahits = np.reshape(np.array(datahits, dtype=float), (Nruns, Ndoms))/oversampling
    datatimesbinned = np.reshape(np.array(datatimesbinned, dtype=float), (Nruns, Ndoms))/oversampling

    return datahits, np.array(datatimes, dtype=float), datatimesbinned, datapositions

def simulate_positions(module, Nobs, N, bounds, radius, d, dN, pa, ps, seed=666, Nthreads=64):
    
    Nphotons = Nobs*N
    print "Total Threads: %s" % Nphotons
    assert(Nphotons <= 1.1e8)

    d = np.uint32(d)
    dN = np.uint32(dN)
    radius = np.float32(radius)
    Ndoms = np.uint32(pow((d/dN)*2+1, 3) - 1)

    t1 = time.time()
    rng_states = get_rng_states(module, Nphotons, seed=seed)
    t2 = time.time()

    d_list = get_doms(module, Ndoms, radius, d, dN)
    t3 = time.time()

    x = np.random.uniform(bounds[0][0], bounds[0][1], N)
    y = np.random.uniform(bounds[1][0], bounds[1][1], N)
    z = np.random.uniform(bounds[2][0], bounds[2][1], N)
    # print x
    # print y
    # print z
    pInit = np.concatenate([x, y, z]).astype(np.float32)
    t4 = time.time()
    
    # print "t2-t1: ", t2-t1
    # print "t3-t2: ", t3-t2
    # print "t4-t3: ", t4-t3

    start = time.time()
    datahits = np.zeros(Ndoms*N, dtype=np.int32)
    datahitsNum = -np.ones(Nobs*N, dtype=np.int32)
    datatimes = np.zeros(Nphotons, dtype=np.float32)
    datapositions = np.zeros(Nphotons*3, dtype=np.float32)
    simulate = module.get_function('simulate_positions')

    simulate(np.uint64(Nphotons), np.uint64(Nobs), rng_states, d_list, cuda.InOut(datahits), cuda.InOut(datatimes), cuda.InOut(datahitsNum), 
             cuda.InOut(datapositions), cuda.In(pInit), np.float32(pa), np.float32(ps),np.uint32(Ndoms), 
             block=(Nthreads, 1, 1), grid=(Nphotons//Nthreads + 1, 1))    

    print "end-start", time.time() - start
    # print "sumHits: ", sum(datahits)
    # print 

    datahits = np.reshape(np.array(datahits, dtype=float), (N, Ndoms))
    datahitsNum = np.reshape(np.array(datahitsNum, dtype=float), (N, Nobs))
    datatimes = np.reshape(np.array(datatimes, dtype=float), (N, Nobs))
    pInit = np.reshape(pInit, (3, N)).T
    return datahits, datahitsNum, datatimes, pInit


def LLH_dima(x, mu, os, deltamu=1e-1, nohit_penalty=False, vectorCalc=True) :
    from scipy.special import loggamma as lgamma
    x = x.copy()
    mu = mu.copy()
    '''
    Calculate dima LLH
    '''
    llh = 0
    if vectorCalc:
        mu_dima = (os*mu+x)/(os+1)
        mask_mu = mu != 0
        mask_x = x != 0
        llh += np.sum(os*mu[mask_mu]*np.log(mu_dima[mask_mu]/mu[mask_mu]))
        llh += np.sum(x[mask_x]*np.log(mu_dima[mask_x]/x[mask_x]))
        if nohit_penalty:
            # print "lel: ", np.sum(x[~mask_mu] * np.log(deltamu) - mu[~mask_mu] - lgamma(x[~mask_mu] + 1))
            llh += np.sum(x[~mask_mu] * np.log(deltamu) - mu[~mask_mu] - lgamma(x[~mask_mu] + 1))
    else:
        for i in range(len(x)):
            mu_dima = (os*mu[i]+x[i])/(os+1)
            if mu[i] != 0:
                llh += os*mu[i]*np.log(mu_dima/mu[i])

            if x[i] != 0:
                llh += x[i]*np.log(mu_dima/x[i])

    return -llh


def LLH_poisson(x, mu, deltamu=1e-1, vectorCalc=True):
    from scipy.special import loggamma as lgamma
    llh = 0
    if vectorCalc:
        mask_mu = mu != 0
        llh += np.sum(x[mask_mu]*np.log(mu[mask_mu]) - mu[mask_mu] - lgamma(x[mask_mu] + 1))
        llh += np.sum(x[~mask_mu] * np.log(deltamu) - mu[~mask_mu] - lgamma(x[~mask_mu] + 1))
    else:
        for i in range(len(x)):
            if mu[i] > 0:
                llh += x[i] * np.log(mu[i]) - mu[i] - lgamma(x[i] + 1)
            else:
                llh += x[i] * np.log(deltamu) - mu[i] - lgamma(x[i] + 1)
    return -llh



# def simulate_photons_fast(Nphotons, d_list, pInit, datahits, datatimes, radius, d, dN, x, y, z, pa, ps, seed=666, Nthreads=64):
#     print d_list
#     d = np.uint32(d)
#     dN = np.uint32(dN)
#     radius = np.float32(radius)
#     Ndoms = np.uint32(pow((d/dN)*2+1, 3) - 1)
    
#     t1 = time.time()
#     rng_states = get_rng_states(Nphotons, seed=seed)
#     t2 = time.time()
#     module = pycuda.compiler.SourceModule(kernel_code, no_extern_c=True, 
#        include_dirs=['/home/thomas/Documents/toy-mc/photon_prob/cuda_tools'])
    
#     simulate = module.get_function('simulate')

#     simulate(np.uint64(Nphotons), rng_states, d_list, cuda.InOut(datahits), cuda.InOut(datatimes), pInit, 
#              np.float32(pa), np.float32(ps),np.int32(Ndoms), 
#              block=(Nthreads, 1, 1), grid=(Nphotons//Nthreads + 1, 1))    
#     t3 = time.time()
    
#     print "t2-t1: ", t2-t1
#     print "t3-t2: ", t3-t2

    
#     return np.array(datahits, dtype=float), np.array(datatimes, dtype=float)
