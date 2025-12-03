import numpy as np
from scipy.integrate import solve_ivp
import gymnasium as gym
from gymnasium.spaces import Box

mu = 3.986e14
n = np.sqrt(mu / (6_678_000)**3)

a2 = np.array([
    [3*n**2, 0, 0, 0, 2*n, 0],
    [0, 0, 0, -2*n, 0, 0],
    [0, 0, -n**2, 0, 0, 0]
])

A = np.block([
    [np.zeros((3,3)), np.eye(3)],
    [a2]
])

B = np.block([[np.zeros((3,3))], [np.eye(3)]])

def dynamics(t, x, u):
    # CWH equations
    return (A@x.reshape(-1,1) + B@u.reshape(-1,1)).ravel()

def observation(x):
    # Measurement noise for partial observability
    # Noise is mean zero gaussian, covariance scales with relative distance
    return x + np.sqrt(np.linalg.norm(x[:3]))/60 * np.random.randn(6)
    

class DockingEnvironment(gym.Env):
    
    def __init__(self):
        self.x = np.zeros(6)    # Starting state
        self.dist0 = 0
        
        self.observation_space = Box(np.array([0., 0., 0., -100., -100., -100.], dtype=np.float64), np.array([1000., 1000., 1000., 100., 100., 100.], dtype=np.float64), dtype=np.float64)
        self.starting_space = Box(np.array([100., 100., 100., -10., -10., -10.], dtype=np.float64), np.array([1000., 1000., 1000., 10., 10., 10.], dtype=np.float64), dtype=np.float64)
        self.action_space = Box(-np.ones(3, dtype=np.float64), np.ones(3, dtype=np.float64), dtype=np.float64)
        
    def reset(self, seed=None):
        self.x = self.starting_space.sample()
        self.dist0 = np.linalg.norm(self.x[:3])
        obs = observation(self.x)
        
        return obs, dict()
    
    def step(self, action):
        action = action*30
        
        # Propagate
        sol = solve_ivp(dynamics, [0,.1], self.x, args=(action,))
        self.x = sol.y[:,-1]
        obs = observation(self.x)
        
        # Get reward
        rew, term, trunc = self.reward(action)
        
        return obs, rew, term, trunc, dict()
    
    def reward(self, action):
        
        # Check conical constraint
        incone = np.all(self.x[:3] >= 0)
        
        # Check terminal condition
        inradius = (np.linalg.norm(self.x[:3]) < 1)
        invel = (np.linalg.norm(self.x[3:]) < .1)
        term = True
        trunc = True
        dist = np.linalg.norm(self.x[:3])
        
        # Terminate if left cone
        if not incone:
            #dist = np.linalg.norm(self.x[:3])
            rew = -1
            
        # Win if in range and moving slow enough
        elif inradius and invel:
            trunc = False
            rew = 1
            
        # Terminate if crash
        elif inradius and not invel:
            rew = -1
            
        else:
            term = False
            trunc = False
            rew = .01 * (self.dist0 - dist) / self.dist0
            #rew = -.001 * np.linalg.norm(action) / np.sqrt(3*30**2)
            
        return rew, term, trunc
            