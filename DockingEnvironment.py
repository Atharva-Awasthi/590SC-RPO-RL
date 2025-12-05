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
    
    def __init__(self, radius=2, velthresh=1):
        self.x = np.zeros(6)    # Starting state
        self.dist0 = 0
        
        self.radius = radius
        self.velthresh = velthresh
        
        self.observation_space = Box(np.array([0., 0., 0., -10., -10., -10.], dtype=np.float64), np.array([100., 100., 100., 10., 10., 10.], dtype=np.float64), dtype=np.float64)
        self.starting_space = Box(np.array([0., 0., 0., -1., -1., -1.], dtype=np.float64), np.array([50., 50., 50., 1., 1., 1.], dtype=np.float64), dtype=np.float64)
        self.action_space = Box(-np.ones(3, dtype=np.float64), np.ones(3, dtype=np.float64), dtype=np.float64)
        
    def reset(self, seed=None):
        self.x = self.starting_space.sample()
        self.dist0 = np.linalg.norm(self.x[:3])
        obs = observation(self.x)
        
        return obs, dict()
    
    def step(self, action):
        
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
        dist = np.linalg.norm(self.x[:3])
        vel = np.linalg.norm(self.x[3:])
        dist_rew = self.radius / dist
        vel_rew = self.velthresh / vel
        term = True
        trunc = True
        
        # Terminate if left cone
        if not incone:
            #dist = np.linalg.norm(self.x[:3])
            print(1)
            rew = -1 + 2*dist_rew
            
        # Terminate if out of bounds
        elif dist > 173.2:
            print(2)
            rew = -2
            
        # Win if in range and moving slow enough
        elif dist <= self.radius and vel <= self.velthresh:
            print(3)
            trunc = False
            rew = 1
            
        # Terminate if crash
        elif dist <= self.radius and vel > self.velthresh:
            print(4)
            rew = 0 + vel_rew
            
        else:
            term = False
            trunc = False
            #rew = -1e-2* np.dot(self.x[:3], self.x[3:]) / np.linalg.norm(self.x[:3]) / np.linalg.norm(self.x[3:])
            rew = 0
            
            
        return rew, term, trunc
            