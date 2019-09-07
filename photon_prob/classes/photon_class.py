#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 09:12:09 2018

@author: thomas
"""

#%%
# Importing packages
import os
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import norm
#%%

class Photon():
    
    def __init__(self, initial_position=(0.0,0.0), initial_direction=0.0, PA=0.001, PS=0.01, dom_pos=[], dom_radii=[]):
        self.P_scat = PS
        self.P_absorb = PA
        self.speed = 1.0                                                       # abitrary units
        if initial_direction == "random":
            initial_direction = np.random.uniform(0.0, np.pi*2.0)
        self.direction = initial_direction                                     # in radians
        self.positions = [np.array(initial_position)]
        self.is_absorbed = False
        self.distance_travelled = 0.0
        self.time_alive = 0
        self.count = 0
        self.scatter_directions = []
        self.scatter_distances = []
        self.scatter_positions = []
        self.prev_scat_pos = initial_position
        self.hit_DOM = 0
        self.dom_pos = dom_pos
        self.dom_radii = dom_radii
        
    def velocity(self):
        vx = np.cos(self.direction)*self.speed
        vy = np.sin(self.direction)*self.speed
        return np.array((vx, vy))
        
    # Determining whether a photons scatters at a given timestep and changes its direction if True
    def scatter(self):
        val = np.random.rand()
        if val < self.P_scat:
            direction = np.random.uniform(0.0, np.pi*2.0)
            self.direction = direction
            self.scatter_directions.append(direction)
            self.scatter_distances.append(self.count)
            self.count = 0
        
    # Determining whether a photons is getting absorbed at a given timestep
    def absorb(self, limit=False):
        val = np.random.rand()
        if val < self.P_absorb or limit:
            self.is_absorbed = True
            self.calc_displacement()
    
    # Calculating displacement of the photon     
    def calc_displacement(self):
        p1 = self.positions[0]
        p2 = self.positions[-1]
        distance = p2-p1
        self.displacement = np.sqrt(distance[0]**2+distance[1]**2)
    
    # Moving a photon a single timestep and the direction given by dv
    def move_photon(self, dt, DOMs=[]):
        distance = dt*self.velocity()
        new_position = self.positions[-1] + distance
        self.positions.append(new_position)
        self.distance_travelled += np.sqrt(distance[0]**2+distance[1]**2)
        
        diff = self.dom_pos-new_position
        dist = norm(diff, axis=1)

        is_hit = np.where((dist <= self.dom_radii) == 1)[0]
        if is_hit.size != 0:
            hit_dom = DOMs[is_hit[0]]
#            print hit_dom
            hit_dom.photon_hits += 1
            self.hit_DOM = 1
            self.is_absorbed = True
            self.calc_displacement()
#        for dom in DOMs:
#            hit = dom.is_hit(new_position)
#            if hit:
#                self.hit_DOM = 1
#                self.is_absorbed = True
#                self.calc_displacement()
                
        self.absorb()
        self.scatter()
        self.count += 1


class DOM():
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.photon_hits = 0
        
    # Checking wheter point is within one of the DOMs
    def is_hit(self, pos):
        distance = (pos - self.center)**2
        distance = np.sqrt(np.sum(distance))
        if distance <= self.radius:
            self.photon_hits += 1
            return True
        return False



def create_doms(xlim, ylim, dx, dy, r):
    xs = np.arange(xlim[0], xlim[1]+0.01, dx)
    ys = np.arange(ylim[0], ylim[1]+0.01, dy)

    DOMs = []
    for x in xs:
        for y in ys:
            if x != 0 or y != 0:
                DOMs.append([[x, y], r])

    return DOMs


class Photon_Simulation():
    
    def __init__(self, N_photons, initial_position=(0.0,0.0), initial_direction=0.0, DOM_attributes=[], 
                 PA=0.001, PS=0.01, time=0, verbose=True, dt=1):
        
        self.N_photons = N_photons
        self.dt = dt
        self.time = time
        self.DOMs = []
        self.dists = []
        self.displs = []
        self.times = []
        self.verbose = verbose
        self.dom_radii = []
        self.dom_pos = []
        for attr in DOM_attributes:
            center, radius = attr
            self.DOMs.append(DOM(center, radius))
            self.dom_radii.append(radius)
            self.dom_pos.append(center)
        
        params = [initial_position, initial_direction, PA, PS, self.dom_pos, self.dom_radii]
        self.photons = [Photon(*params) for _ in range(N_photons)]
    
    
    # Simulate the path of N photons. Returning all the positions and the distance they have traveled
    def simulate_photons(self):
        N = self.N_photons
        dN = N//20
        for i, photon in enumerate(self.photons):
            if self.verbose and N > 20 and (i+1)%dN == 0:
                print "\r {} out of {} photons have been simulated".format(i+1, self.N_photons),
            c = 0
            while not photon.is_absorbed:
                if photon.P_absorb == 0 and c > 10000:
                    photon.absorb(limit=True)
                    break
                photon.move_photon(self.dt, self.DOMs)
                c += 1
            self.dists.append(photon.distance_travelled)
            self.displs.append(photon.displacement)
            self.times.append(c*self.dt)


    def get_DOM_hits(self):
        hits = []
        for dom in self.DOMs:
            hits.append(dom.photon_hits)
        return np.array(hits).astype(float)

    # Plotting the photons paths and the distribution of distances they have traveled
    def plot_photons(self, grid=None):
        fig, ax = plt.subplots(figsize=(14,14))
        color_list = plt.cm.tab20c(np.linspace(0, 1, self.N_photons))
        dists = []
        
        for i, photon in enumerate(self.photons):
            positions = np.array(photon.positions)
            ax.plot(positions[:,0], positions[:,1], color=color_list[i])
            dists.append(photon.distance_travelled)
        for dom in self.DOMs:
            c = plt.Circle(dom.center, dom.radius, fc="white", ec="blue")
            ax.add_artist(c)
            ax.text(dom.center[0]-0.5, dom.center[1]-0.5, dom.photon_hits, fontsize=12)

        if grid is not None:
            ax.set_xlim(grid[0][0], grid[0][1])
            ax.set_ylim(grid[1][0], grid[1][1])
#

#        ax2.hist(dists)
#        ax2.set_xlabel("Distance")
#        ax2.set_ylabel(r"$N_{photons}$")
#        ax2.set_title("Distance Travelled")

    def save_to_file(self, filename="data/photon_stats.csv", overwrite=False):
        
        if not os.path.isfile(filename) or overwrite:
            d = self.dists
            s = self.displs
            t = self.times
    
            data = np.array([d, s, t]).T
            
            df = pd.DataFrame(data, columns=["distance", "displacement", "time"])
            df.to_csv(filename)
        else:
            print "File with this name already exists and therefor this data has not been saved. Add 'overwrite=True' if overwriting the file is desired."
            
    def save_to_pickle(self, filename="data/photon_stats.p", overwrite=False):
        
        if not os.path.isfile(filename) or overwrite:
            
            d = self.dists
            s = self.displs
            t = self.times
            
            scatter_directions = []
            scatter_distances = []
            hit_DOM = []
            for photon in self.photons:
                scatter_directions += photon.scatter_directions
                scatter_distances += photon.scatter_distances
                hit_DOM.append(photon.hit_DOM)
            
            scatter_distances = np.array(scatter_distances)*self.dt
            
            data = [d, s, t, scatter_directions, scatter_distances, hit_DOM]            
            pickle.dump( data, open( filename, "wb" ) )
            
        else:
            print "File with this name already exists and therefor this data has not been saved. Add 'overwrite=True' if overwriting the file is desired."

#%%

class Track_Simulation():
    
    def __init__(self, initial_direction, P_cascade, cascade_params, max_time=200):
        if initial_direction == "random":
            initial_direction = np.random.uniform(0.0, np.pi*2.0)
        self.direction = initial_direction    
        self.cascade_params = cascade_params
        self.P_cascade = P_cascade
        self.max_time = max_time
        self.time = 0
        self.speed = 1
        self.dt = 1
        self.initial_position = np.array([0,0])
        self.cascades = []
        self.positions = [self.initial_position]
        self.distance_travelled = 0

    def simulate(self):
        while self.time <= self.max_time:
            self.move_particle()
            val = np.random.rand()
            if val < self.P_cascade:
                p = self.get_cascade_params()
                sim = Photon_Simulation(*p)
                sim.simulate_photons()
                self.cascades.append(sim)
    
    def get_cascade_params(self):
        p = self.cascade_params
        p[0] = int(np.random.normal(p[0], 2))
        p[1] = self.positions[-1]
        p[6] = self.time
        return p
    
    def move_particle(self):
        self.time += 1
        distance = self.dt*self.velocity()
        new_position = self.positions[-1] + distance
        self.positions.append(new_position)
        self.distance_travelled += np.sqrt(distance[0]**2+distance[1]**2)

    def velocity(self):
        vx = np.cos(self.direction)*self.speed
        vy = np.sin(self.direction)*self.speed
        return np.array((vx, vy))
    
    #%% Animations related functions
    
#    def run_animation(positions_all):
#        
#        def init():
#            for line in lines:
#                line.set_data([], [])
#            return lines
#    
#        def data_gen():
#            cnt = 0
#            while cnt < maxlen:
#                cnt += 5
#                xs = []
#                ys = []
#                for positions in positions_all:
#                    xs.append(positions[:cnt-1,0])
#                    ys.append(positions[:cnt-1, 1])
#        
#                yield xs, ys
#        
#        def run(data):
#            # update the data
#            xs, ys = data
#            for line, xdata, ydata in zip(lines, xs, ys):
#                line.set_data(xdata, ydata)
#            return lines
#        
#        def calculate_sizes(positions_all):
#            xmins, xmaxs, ymins, ymaxs, lengths = [], [], [], [], []
#            for positions in positions_all:
#                xmins.append(min(positions[:, 0]))
#                xmaxs.append(max(positions[:, 0]))
#                ymins.append(min(positions[:, 1]))
#                ymaxs.append(max(positions[:, 1]))
#                lengths.append(len(positions))
#            
#            return min(xmins)-1, max(xmaxs)+1, min(ymins)-1, max(ymaxs)+1, max(lengths)
#        
#        fig, ax = plt.subplots()
#        circles = initialize_DOMs()
#        for c in circles:
#            ax.add_artist(c)
#        ax.grid()
#        color_list = plt.cm.tab20c(np.linspace(0, 1, len(positions_all)))
#        lines = []
#        for index in range(len(positions_all)):
#            lobj = ax.plot([],[],lw=2,color=color_list[index])[0]
#            lines.append(lobj)
#    
#        xmin, xmax, ymin, ymax, maxlen = calculate_sizes(positions_all)       
#        ax.set_ylim(ymin, ymax)
#        ax.set_xlim(xmin, xmax)
#        
#        ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10,
#                                      repeat=False, init_func=init)
#        return ani




