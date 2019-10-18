#include <iostream>
#include <time.h>
#include "float.h"
#include <math.h>

#include <curand_kernel.h>
#include "stdio.h"

#include "vec3.h"
#include "dom_RT.h"

extern "C"
{

__global__ void init_rng(const unsigned long long int nthreads, curandStatePhilox4_32_10_t *rand_states, 
                         unsigned long long seed) 
{

    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id <= nthreads) {
        curand_init(seed, id, 0, &rand_states[id]);
        // printf("seed: %(PRINT_L)s \\n", seed);
        // printf("id: %(PRINT_I)s \\n", id);
        //curandStatePhilox4_32_10_t* s = new curandStatePhilox4_32_10_t;
        //if (s != 0) {
        //    curand_init(seed, id, 0, s);
        //}
        //rand_states[id] = *s;
    }

}

__global__ void init_times(const unsigned long long int nthreads, float *times)
{
        int id = blockIdx.x*blockDim.x + threadIdx.x;

        if (id >= nthreads)
                return;

        times[id] = 0.0f;
}



__global__ void init_pInit(const unsigned long long int nthreads, float x, float y, float z, vec3 *pInit)
{
        int id = blockIdx.x*blockDim.x + threadIdx.x;

        if (id >= nthreads)
                return;
        // printf("Initiating!! \\n");
        *pInit = vec3(x, y, z);
}


__global__
void create_doms(float radius, int d, int dN, dom **d_list){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int c = 0;
        for (int i=-d; i<=d; i += dN) {
            for (int j=-d; j<=d; j += dN) {
                for (int k=-d; k<=d; k += dN) {
                    if (i != 0 || j != 0 || k != 0) {
                        d_list[c++] = new dom(vec3( i,  j,  k), radius);
                        // hits[c++] = 0;
                    }
                }
            }
        }

    }
}

__device__
vec3 random_direction_3d(curandStatePhilox4_32_10_t *local_rand_state) {
    float dir_theta = curand_uniform(local_rand_state)*2.0f*M_PI;
    float vz = 2.0f*curand_uniform(local_rand_state)-1.0f;
    float vx = sqrtf(1.0f-powf(vz,2))*cosf(dir_theta);
    float vy = sqrtf(1.0f-powf(vz,2))*sinf(dir_theta);            
    return vec3(vx, vy, vz);
}


__device__
float sample_interaction(float p, curandStatePhilox4_32_10_t *local_rand_state) {
    float ranNum = curand_uniform(local_rand_state);
    return -logf(ranNum)*p;
}

__global__
void simulate(const unsigned long long int nthreads, const unsigned long long int nobs, curandStatePhilox4_32_10_t *rand_states, dom **d_list, 
              int *hits, float *times, float *timesbinned, float *positions, vec3 *pInit, float pa, float ps, int Ndoms) {

    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind >= nthreads) return;

    int run = ind/nobs;
    // unsigned long long int indR = ind + (run * nobs);
    // printf("ind: %(PRINT_I)s, run: %(PRINT_I)s \n", ind, run);

    // if (ind == nthreads-1) printf("-------------------------------------------------------\n");

    curandStatePhilox4_32_10_t local_rand_state = rand_states[ind];
    vec3 dir = random_direction_3d(&local_rand_state);

    vec3 pHit;
    ray r(pInit[0], dir);
    vec3 Ppos = *pInit;

    float pDist = sample_interaction(pa, &local_rand_state);
    float scatter = sample_interaction(ps, &local_rand_state);
    
    // printf("pDist: %(PRINT_F)s \\n", pDist);
    // printf("scatter: %(PRINT_F)s \\n", scatter);
    
    float t = 0;
    float tHit;
    while (scatter < pDist) {
        for (int k=0; k<Ndoms; k++){
            if ((*d_list[k]).hit(r, 0, scatter, &pHit, &tHit)) {
                atomicAdd(&hits[k + Ndoms*run], 1);
                atomicAdd(&timesbinned[k + Ndoms*run], t + tHit);
                times[ind] = t + tHit;
                positions[ind] = pHit.x();
                positions[ind + nthreads ] = pHit.y();
                positions[ind + nthreads*2] = pHit.z();
                // printf("ind: %(PRINT_I)s, domID: %(PRINT_I)s, time %(PRINT_F)s, x[%(PRINT_I)s]: %(PRINT_F)s, y[%(PRINT_I)s]: %(PRINT_F)s, z[%(PRINT_I)s]: %(PRINT_F)s \n", 
                        // ind, k, times[ind], ind, pHit.x(), ind + nthreads , pHit.y(), ind + nthreads*2 , pHit.z());
                return;
            }
        }
        t += scatter;
        Ppos += r.direction()*scatter;
        pDist -= scatter;
        r = ray(Ppos, random_direction_3d(&local_rand_state));
        scatter = sample_interaction(ps, &local_rand_state);
    }


    for (int k=0; k<Ndoms; k++){
        if ((*d_list[k]).hit(r, 0, pDist, &pHit, &tHit)) {
            atomicAdd(&hits[k + Ndoms*run], 1);
            atomicAdd(&timesbinned[k + Ndoms*run], t + tHit);
            times[ind] = t + tHit;
            positions[ind] = pHit.x();
            positions[ind + nthreads ] = pHit.y();
            positions[ind + nthreads*2 ] = pHit.z();
            // printf("ind: %(PRINT_I)s, domID: %(PRINT_I)s, time %(PRINT_F)s, x[%(PRINT_I)s]: %(PRINT_F)s, y[%(PRINT_I)s]: %(PRINT_F)s, z[%(PRINT_I)s]: %(PRINT_F)s \n", 
                    // ind, k, times[ind], ind, pHit.x(), ind + nthreads , pHit.y(), ind + 2*nthreads , pHit.z());
            return;
        }
    }
    const char type = '0';
    positions[ind] = nanf(&type);
    positions[ind + nthreads ] = nanf(&type);
    positions[ind + nthreads*2 ] = nanf(&type);
    // printf("NO HIT: ind: %(PRINT_I)s, domID: %(PRINT_I)s, time %(PRINT_F)s, x[%(PRINT_I)s]: %(PRINT_F)s, y[%(PRINT_I)s]: %(PRINT_F)s, z[%(PRINT_I)s]: %(PRINT_F)s \n", 
            // ind, -1, times[ind], ind, -1.0f, ind + nthreads , -1.0f, ind + 2*nthreads , -1.0f);

}

__global__
void simulate_grid(const unsigned long long int nthreads, const unsigned long long int nobs, curandStatePhilox4_32_10_t *rand_states, dom **d_list, 
              int *hits, float *times, float *timesbinned, float *positions, float *pInit, float pa, float ps, int Ndoms) {

    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind >= nthreads) return;

    int run = ind/nobs;
    int runs = nthreads/nobs;
    // unsigned long long int indR = ind + (run * nobs);
    // printf("ind: %(PRINT_I)s, run: %(PRINT_I)s \n", ind, run);

    // if (ind == nthreads-1) printf("-------------------------------------------------------\n");

    curandStatePhilox4_32_10_t local_rand_state = rand_states[ind];
    vec3 dir = random_direction_3d(&local_rand_state);

    vec3 pHit;
    float x = pInit[run];
    float y = pInit[run + runs];
    float z = pInit[run + 2*runs];
    vec3 Ppos = vec3(x, y, z);
    ray r(Ppos, dir);

    float pDist = sample_interaction(pa, &local_rand_state);
    float scatter = sample_interaction(ps, &local_rand_state);
    
    // printf("pDist: %(PRINT_F)s \\n", pDist);
    // printf("scatter: %(PRINT_F)s \\n", scatter);
    
    float t = 0;
    float tHit;
    while (scatter < pDist) {
        for (int k=0; k<Ndoms; k++){
            if ((*d_list[k]).hit(r, 0, scatter, &pHit, &tHit)) {
                atomicAdd(&hits[k + Ndoms*run], 1);
                atomicAdd(&timesbinned[k + Ndoms*run], t + tHit);
                // printf("hitTime: %(PRINT_F)s  \n", t + tHit);
                times[ind] = t + tHit;
                positions[ind] = pHit.x();
                positions[ind + nthreads ] = pHit.y();
                positions[ind + nthreads*2] = pHit.z();
                // printf("ind: %(PRINT_I)s, domID: %(PRINT_I)s, time %(PRINT_F)s, x[%(PRINT_I)s]: %(PRINT_F)s, y[%(PRINT_I)s]: %(PRINT_F)s, z[%(PRINT_I)s]: %(PRINT_F)s \n", 
                        // ind, k, times[ind], ind, pHit.x(), ind + nthreads , pHit.y(), ind + nthreads*2 , pHit.z());
                return;
            }
        }
        t += scatter;
        Ppos += r.direction()*scatter;
        pDist -= scatter;
        r = ray(Ppos, random_direction_3d(&local_rand_state));
        scatter = sample_interaction(ps, &local_rand_state);
    }


    for (int k=0; k<Ndoms; k++){
        if ((*d_list[k]).hit(r, 0, pDist, &pHit, &tHit)) {
            atomicAdd(&hits[k + Ndoms*run], 1);
            atomicAdd(&timesbinned[k + Ndoms*run], t + tHit);
            // printf("hitTime: %(PRINT_F)s \n", t + tHit);
            times[ind] = t + tHit;
            positions[ind] = pHit.x();
            positions[ind + nthreads ] = pHit.y();
            positions[ind + nthreads*2 ] = pHit.z();
            // printf("ind: %(PRINT_I)s, domID: %(PRINT_I)s, time %(PRINT_F)s, x[%(PRINT_I)s]: %(PRINT_F)s, y[%(PRINT_I)s]: %(PRINT_F)s, z[%(PRINT_I)s]: %(PRINT_F)s \n", 
                    // ind, k, times[ind], ind, pHit.x(), ind + nthreads , pHit.y(), ind + 2*nthreads , pHit.z());
            return;
        }
    }
    const char type = '0';
    positions[ind] = nanf(&type);
    positions[ind + nthreads ] = nanf(&type);
    positions[ind + nthreads*2 ] = nanf(&type);
    // printf("NO HIT: ind: %(PRINT_I)s, domID: %(PRINT_I)s, time %(PRINT_F)s, x[%(PRINT_I)s]: %(PRINT_F)s, y[%(PRINT_I)s]: %(PRINT_F)s, z[%(PRINT_I)s]: %(PRINT_F)s \n", 
            // ind, -1, times[ind], ind, -1.0f, ind + nthreads , -1.0f, ind + 2*nthreads , -1.0f);

}


__global__
void simulate_positions(const unsigned long long int nthreads, const unsigned long long int nobs, curandStatePhilox4_32_10_t *rand_states, dom **d_list, 
              int *hits, float *times, int *hitsNum, float *positions, float *pInit, float pa, float ps, int Ndoms) {

    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind >= nthreads) return;

    int run = ind/nobs;
    int runs = nthreads/nobs;
    // unsigned long long int indR = ind + (run * nobs);
    // printf("ind: %(PRINT_I)s, run: %(PRINT_I)s \n", ind, run);

    // if (ind == nthreads-1) printf("-------------------------------------------------------\n");

    curandStatePhilox4_32_10_t local_rand_state = rand_states[ind];
    vec3 dir = random_direction_3d(&local_rand_state);

    vec3 pHit;
    float x = pInit[run];
    float y = pInit[run + runs];
    float z = pInit[run + 2*runs];
    vec3 Ppos = vec3(x, y, z);
    ray r(Ppos, dir);

    float pDist = sample_interaction(pa, &local_rand_state);
    float scatter = sample_interaction(ps, &local_rand_state);
    
    // printf("pDist: %(PRINT_F)s \\n", pDist);
    // printf("scatter: %(PRINT_F)s \\n", scatter);
    
    float t = 0;
    float tHit;
    while (scatter < pDist) {
        for (int k=0; k<Ndoms; k++){
            if ((*d_list[k]).hit(r, 0, scatter, &pHit, &tHit)) {
                atomicAdd(&hits[k + Ndoms*run], 1);
                hitsNum[ind] = k;
                // printf("hitTime: %(PRINT_F)s  \n", t + tHit);
                times[ind] = t + tHit;
                positions[ind] = pHit.x();
                positions[ind + nthreads ] = pHit.y();
                positions[ind + nthreads*2] = pHit.z();
                // printf("ind: %(PRINT_I)s, domID: %(PRINT_I)s, time %(PRINT_F)s, x[%(PRINT_I)s]: %(PRINT_F)s, y[%(PRINT_I)s]: %(PRINT_F)s, z[%(PRINT_I)s]: %(PRINT_F)s \n", 
                        // ind, k, times[ind], ind, pHit.x(), ind + nthreads , pHit.y(), ind + nthreads*2 , pHit.z());
                return;
            }
        }
        t += scatter;
        Ppos += r.direction()*scatter;
        pDist -= scatter;
        r = ray(Ppos, random_direction_3d(&local_rand_state));
        scatter = sample_interaction(ps, &local_rand_state);
    }


    for (int k=0; k<Ndoms; k++){
        if ((*d_list[k]).hit(r, 0, pDist, &pHit, &tHit)) {
            atomicAdd(&hits[k + Ndoms*run], 1);
            hitsNum[ind] = k;
            // printf("hitTime: %(PRINT_F)s \n", t + tHit);
            times[ind] = t + tHit;
            positions[ind] = pHit.x();
            positions[ind + nthreads ] = pHit.y();
            positions[ind + nthreads*2 ] = pHit.z();
            // printf("ind: %(PRINT_I)s, domID: %(PRINT_I)s, time %(PRINT_F)s, x[%(PRINT_I)s]: %(PRINT_F)s, y[%(PRINT_I)s]: %(PRINT_F)s, z[%(PRINT_I)s]: %(PRINT_F)s \n", 
                    // ind, k, times[ind], ind, pHit.x(), ind + nthreads , pHit.y(), ind + 2*nthreads , pHit.z());
            return;
        }
    }
    const char type = '0';
    positions[ind] = nanf(&type);
    positions[ind + nthreads ] = nanf(&type);
    positions[ind + nthreads*2 ] = nanf(&type);
    // printf("NO HIT: ind: %(PRINT_I)s, domID: %(PRINT_I)s, time %(PRINT_F)s, x[%(PRINT_I)s]: %(PRINT_F)s, y[%(PRINT_I)s]: %(PRINT_F)s, z[%(PRINT_I)s]: %(PRINT_F)s \n", 
            // ind, -1, times[ind], ind, -1.0f, ind + nthreads , -1.0f, ind + 2*nthreads , -1.0f);

}


__global__
void simulate_fast(const unsigned long long int nthreads, curandStatePhilox4_32_10_t *rand_states, dom **d_list, 
              int *hits, float *pInit, int run, float pa, float ps, int Ndoms) {

    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind >= nthreads) return;

    // unsigned long long int indR = ind + (run * nobs);
    // printf("ind: %(PRINT_I)s, run: %(PRINT_I)s \n", ind, run);

    // if (ind == nthreads-1) printf("-------------------------------------------------------\n");

    curandStatePhilox4_32_10_t local_rand_state = rand_states[ind + run*nthreads];
    vec3 dir = random_direction_3d(&local_rand_state);

    // printf("dirx %(PRINT_F)s, diry %(PRINT_F)s, dirz %(PRINT_F)s\n", dir.x(), dir.y(), dir.z());

    vec3 pHit;
    float x = pInit[0];
    float y = pInit[1];
    float z = pInit[2];
    vec3 Ppos = vec3(x, y, z);
    ray r(Ppos, dir);

    float pDist = sample_interaction(pa, &local_rand_state);
    float scatter = sample_interaction(ps, &local_rand_state);
    
    // printf("pDist: %(PRINT_F)s \\n", pDist);
    // printf("scatter: %(PRINT_F)s \\n", scatter);
    
    float t = 0;
    float tHit;
    while (scatter < pDist) {
        for (int k=0; k<Ndoms; k++){
            if ((*d_list[k]).hit(r, 0, scatter, &pHit, &tHit)) {
                atomicAdd(&hits[k], 1);
                // printf("hitTime: %(PRINT_F)s  \n", t + tHit);
                // printf("ind: %(PRINT_I)s, domID: %(PRINT_I)s, time %(PRINT_F)s, x[%(PRINT_I)s]: %(PRINT_F)s, y[%(PRINT_I)s]: %(PRINT_F)s, z[%(PRINT_I)s]: %(PRINT_F)s \n", 
                // ind, k, times[ind], ind, pHit.x(), ind + nthreads , pHit.y(), ind + nthreads*2 , pHit.z());
                return;
            }
        }
        t += scatter;
        Ppos += r.direction()*scatter;
        pDist -= scatter;
        r = ray(Ppos, random_direction_3d(&local_rand_state));
        scatter = sample_interaction(ps, &local_rand_state);
    }


    for (int k=0; k<Ndoms; k++){
        if ((*d_list[k]).hit(r, 0, pDist, &pHit, &tHit)) {
            atomicAdd(&hits[k], 1);
            // printf("hitTime: %(PRINT_F)s \n", t + tHit);
            // printf("ind: %(PRINT_I)s, domID: %(PRINT_I)s, time %(PRINT_F)s, x[%(PRINT_I)s]: %(PRINT_F)s, y[%(PRINT_I)s]: %(PRINT_F)s, z[%(PRINT_I)s]: %(PRINT_F)s \n", 
                    // ind, k, times[ind], ind, pHit.x(), ind + nthreads , pHit.y(), ind + 2*nthreads , pHit.z());
            return;
        }
    }
    // printf("NO HIT: ind: %(PRINT_I)s, domID: %(PRINT_I)s, time %(PRINT_F)s, x[%(PRINT_I)s]: %(PRINT_F)s, y[%(PRINT_I)s]: %(PRINT_F)s, z[%(PRINT_I)s]: %(PRINT_F)s \n", 
            // ind, -1, times[ind], ind, -1.0f, ind + nthreads , -1.0f, ind + 2*nthreads , -1.0f);

}

} 