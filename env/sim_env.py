import scipy.stats as stats

import numpy as np

# new env with piecewise linear profile for dosage effect
class newEnv:
    def __init__(self,timeout=10):
        self.timeout = timeout
        self.time = 0
        self.steps_elapsed = 0
        self.threshold = 0.0
        self.scale = 0.5
        self.h0 = 12.5
        self.h = self.h0-12
        self.d_optim = self.get_optim()
        self.m = self.get_obs()

    def reset(self):
        self.time = 0
        self.steps_elapsed = 0
        self.h = self.h0-12
        self.m = self.get_obs()
        self.d_optim = self.get_optim()
        return self.m
    
    def get_optim(self):
        h_diff = self.h0 - self.h
        d_optim = h_diff
        return d_optim

    def get_obs(self):
        if self.h> self.threshold:
            m = stats.norm.rvs(loc=self.h, scale=0.5, size=1)[0]
        else:
            m = self.threshold
        return np.array([m])
    def h_decay(self,dt):
        if self.h>self.threshold:
            death = False
        else:
            death =True
            T_max =0
            return death, T_max

        T_max = (self.h-self.threshold)/self.scale

        if T_max <= dt:
            death = True
            self.h = self.threshold
        else:
            T_max = dt
            self.h = self.h - dt * self.scale
        return death, T_max


    def h_boost(self,dosage):
        #below d_optim
        if dosage<=self.d_optim:
            slope = 1.0
            self.h += slope*dosage
        # above d_optim
        else:
            slope = -1.0
            self.h += slope*(dosage-2*self.d_optim)
        
    def step(self,action):
        dosage = action[0]
        delta_t = action[1]
        self.steps_elapsed+=1
        self.h_boost(dosage)
        death, T_max = self.h_decay(delta_t)
        if death:
            done =True
            reward = T_max
            self.time = T_max + self.time
        else:
            done = False
            reward = delta_t.item()
            self.time = delta_t+self.time
            self.d_optim = self.get_optim()
        self.m = self.get_obs()
        return self.m, reward, done or self.steps_elapsed >= self.timeout, death


