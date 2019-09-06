#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:28:39 2018

@author: thomas
"""

#%%
import numpy as np
from numpy.linalg import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import cdist
#%%
# Plotting parameters
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 15
#%%

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


class photon_ensemble():

    def __init__(self, Nphotons=100, initial_position=np.array([0,0]), initial_direction=0 , dt=1, 
                 PS=0.01, PA=0.01, DOMs=[], verbose=False, time=0):
        self.Nphotons = Nphotons
        self.positions = np.ones((Nphotons, 2))*initial_position
        self.speed = 1
        self.all_positions = [self.positions.copy()]
        self.dt = dt
        self.ps = PS*self.dt
        self.pa = PA*self.dt
        self.absorbed = np.zeros(Nphotons)
        self.time = time
        if DOMs[0] != []:
            self.dom_centers = DOMs[0]
            self.dom_radii = DOMs[1]
            self.dom_hits = np.zeros(len(self.dom_centers))
            self.dom_hit_time = -np.ones(len(self.dom_centers))
            self.noDoms = False
            self.dom_hit_times = [[] for _ in range(len(self.dom_centers))]
        else:
            self.noDoms = True
            
        if initial_direction == "random":
            self.directions = np.random.uniform(0, 2*np.pi, Nphotons)
        else:
            self.directions = np.ones(Nphotons)*initial_direction
        
        self.velocities = self.velocity()
        
    
    def velocity(self, scatters=None):
        # if scatters is None:
        if type(scatters) == type(None):
            vx = np.cos(self.directions)*self.speed
            vy = np.sin(self.directions)*self.speed
        else:
            vx = np.cos(self.directions[scatters])*self.speed
            vy = np.sin(self.directions[scatters])*self.speed
        veloc  = np.array([vx, vy]).T
        return veloc
    
    
    def move_photons(self):
        self.mask = self.absorbed == 0
        self.positions[self.mask] = self.positions[self.mask] + self.velocities[self.mask]*self.dt
        self.all_positions.append(self.positions.copy())
        if not self.noDoms:
            self.check_if_hit()
        self.scatter()
        self.absorb()
        self.time += 1*self.dt
        
    def scatter(self):
        vals = np.random.rand(self.Nphotons)
        scatters = np.where((vals<self.ps) == True)[0]
        if scatters.size > 0:
            size = scatters.size
            self.directions[scatters] = np.random.uniform(0, 2*np.pi, size)
            vs = self.velocity(scatters=scatters)
            self.velocities[scatters] = vs


    def absorb(self):  
        vals = np.random.rand(self.Nphotons)
        absorbs = np.where((vals<self.pa) == True)[0]
        if absorbs.size > 0:
            self.absorbed[absorbs] = 1
            
            
    def check_if_hit(self):
        dist = cdist(self.positions, self.dom_centers)
        hits = np.where(((dist <= self.dom_radii) == 1)*(self.absorbed.reshape(self.absorbed.size,1)==0))
        photon_hit = hits[0]
        dom_hits = hits[1]
        u, c = np.unique(dom_hits, return_counts=True)
        if dom_hits.size !=0:
            self.dom_hits[u] += c
            self.absorbed[photon_hit] = 1
            self.dom_hit_time[u[(self.dom_hit_time == -1)[u]]] = self.time
            for ind, hits in zip(u, c):
                self.dom_hit_times[ind].append([self.time, hits])
    
    def get_positions(self):
        return np.array(self.all_positions)

    def simulate(self):
        while sum(self.absorbed) < self.Nphotons:
            self.move_photons()


    # def plot_cascade(self, savefig=False, center_doms=False):
    #     all_positions = self.get_positions()

    #     fig, ax = plt.subplots(figsize=(17.5,10))
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.grid()
    #     color_list = plt.cm.tab20c(np.linspace(0, 1, all_positions.shape[1]))
    #     for i in range(self.Nphotons):
    #         p1 = all_positions[:, i, :]
    #         ax.plot(p1[:,0], p1[:,1], color=color_list[i])
            
    #     if center_doms:     
    #         ax.set_ylim(-30, 30)
    #         ax.set_xlim(-30, 30)    
        
    #     plt.tight_layout()
    #     if not self.noDoms:
    #         data = self.dom_hits
    #         for dom_center, dom_radius, i in zip(self.dom_centers, self.dom_radii, range(len(self.dom_centers))):
    #             c = plt.Circle(dom_center, dom_radius, fc="white", ec="blue")
    #             ax.add_artist(c)
    #             ax.text(dom_center[0]-0.5, dom_center[1]-0.5, int(data[i]), fontsize=12)
    #     if savefig:
    #         plt.savefig("cascade.eps")

    def plot_photons(self, grid=None):
        fig, ax = plt.subplots(figsize=(14,14))
        color_list = plt.cm.tab20c(np.linspace(0, 1, self.Nphotons))
        dists = []
        positions = self.get_positions()
        positions = np.swapaxes(positions.T, 1, 2)
        ax.plot(*positions)

        # for i, photon in enumerate(self.photons):
        #     positions = np.array(photon.positions)
        #     ax.plot(positions[:,0], positions[:,1], color=color_list[i])
        #     dists.append(photon.distance_travelled)
        for center, radius, hits in zip(self.dom_centers, self.dom_radii, self.dom_hits):
            c = plt.Circle(center, radius, fc="white", ec="blue")
            ax.add_artist(c)
            ax.text(center[0]-0.5, center[1]-0.5, hits, fontsize=12)

        if grid is not None:
            ax.set_xlim(grid[0][0], grid[0][1])
            ax.set_ylim(grid[1][0], grid[1][1])


class track_ensemble():
    
    def __init__(self, Nphotons, params, initial_position=np.array([0,0]), initial_direction=0, Pcascade=0.1, maxtime=200):
        self.positions = [initial_position]
        if initial_direction == "random":
            self.direction = np.random.uniform(0, 2*np.pi)
        else:
            self.direction = initial_direction
        self.Nphotons = Nphotons
        self.dt = params["dt"]
        self.maxtime = maxtime
        self.speed = 1
        self.time = 0
        self.Pcascade = Pcascade
        self.params = params
        self.cascade_positions = []
        self.cascade_times = []
        self.cascade_dom_times = []
        self.cascades = []
        
    def simulate(self):
        while self.time < self.maxtime:
            if np.random.rand() < self.Pcascade:
                Nphotons = abs(int(np.random.normal(self.Nphotons, 5)))+1
                self.params["initial_position"] = self.positions[-1]
                self.params["time"] = self.time
                cascSim = photon_ensemble(Nphotons, **self.params)
                cascSim.simulate()
                self.cascade_positions.append(cascSim.get_positions())
                self.cascade_times.append(self.time)
                self.cascade_dom_times.append(cascSim.dom_hit_times)
                self.cascades.append(cascSim)
                
            new_position = self.positions[-1] + self.dt*self.velocity()
            self.positions.append(new_position)
            self.time += 1*self.dt
            
    
    def velocity(self):
        vx = np.cos(self.direction)*self.speed
        vy = np.sin(self.direction)*self.speed
        veloc  = np.array([vx, vy]).T
        return veloc
        
    def get_positions(self):
        return np.array(self.positions)
    
    def get_animation_params(self):
        casc_positions = self.cascade_positions
        casc_times = self.cascade_times
        positions_all = []
        times = []
        cascade_num = []
        for i, position in enumerate(casc_positions):
            for j in range(position.shape[1]):
                pos = position[:,j,:]
                positions_all.append(pos)
                times.append(casc_times[i])
                cascade_num.append(i)
        
        positions_track = np.array(self.positions)
        positions_all.append(positions_track)
        times.append(0)
    
        Ndoms = len(self.params["DOMs"][0])
        dom_times = [[] for _ in range(Ndoms)]
        for i in range(Ndoms):
            for t in self.cascade_dom_times:
                dom_times[i] += t[i]
    
        return positions_all, times, cascade_num, dom_times

    def run_animation(self, save_animation=False):
        
        def init():
            for line in lines:
                line.set_data([], [])
            return lines
    
        def data_gen():
            cnt = 0
            while cnt < maxlen:
                cnt += 1
                xs = []
                ys = []
                for positions, time in zip(positions_all, times):
                    if cnt-1-time > 0:
                        index = cnt-1-time
                    else:
                        index = 0
                    xs.append(positions[:index, 0])
                    ys.append(positions[:index, 1])
                    
                dom_hits = []
                for dht in dom_times:
                    hit_times = np.array(dht)
                    if hit_times.size != 0:
                        dom_hits.append(hit_times[:,1][hit_times[:,0] <= cnt-3].sum())
                    else:
                        dom_hits.append(0)
        
                yield xs, ys, dom_hits
        
        def run(data):
            # update the data
            xs, ys, dom_hits = data
            for line, xdata, ydata in zip(lines, xs, ys):
                line.set_data(xdata, ydata)
            
            for text, hits in zip(texts, dom_hits):
                text.set_text(str(hits))
                
            return lines, texts
        
        def calculate_sizes(positions_all, times):
            xmins, xmaxs, ymins, ymaxs, lengths = [], [], [], [], []
            for positions, time in zip(positions_all, times):
                xmins.append(min(positions[:, 0]))
                xmaxs.append(max(positions[:, 0]))
                ymins.append(min(positions[:, 1]))
                ymaxs.append(max(positions[:, 1]))
                lengths.append(len(positions)+time)
            
            return min(xmins)-10, max(xmaxs)+10, min(ymins)-10, max(ymaxs)+10, max(lengths)
        
        positions_all, times, cascade_num, dom_times = self.get_animation_params()
        DOMs = self.params["DOMs"]
        
        fig, ax = plt.subplots(figsize=(16,9))
        fig.canvas.manager.window.activateWindow()
        fig.canvas.manager.window.raise_()
        texts = []
        for i in range(len(DOMs[0])):
            c = plt.Circle(DOMs[0][i], DOMs[1][i], fc="white", ec="blue")
            text = plt.text(DOMs[0][i][0]-0.5, DOMs[0][i][1]-0.5, str(0))
            texts.append(text)
            ax.add_artist(c)
                
        ax.grid()
        color_list = plt.cm.tab20c(np.linspace(0, 1, len(np.unique(cascade_num))))
        lines = []
        for index in range(len(positions_all)):
            if index != len(positions_all)-1:
                lobj = ax.plot([],[],lw=2,color=color_list[cascade_num[index]])[0]
                lines.append(lobj)
            else:
                lobj = ax.plot([],[],lw=3,color="black")[0]
                lines.append(lobj)            
    
        xmin, xmax, ymin, ymax, maxlen = calculate_sizes(positions_all, times) 
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        
        ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10,
                                      repeat=False, init_func=init, save_count=maxlen)
        if save_animation:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
            ani.save('track.mp4', writer=writer)
        
        return ani

