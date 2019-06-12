import numpy as np
from physics_sim import PhysicsSim
import sys
import math

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, debug=False):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3 # Good RESULT

        self.state_size = self.action_repeat * 6

        # What are these??
        # Min value of each action dimension
        self.action_low = 0
        # Max value of each action dimension
        self.action_high = 900
        #self.action_high = 1100

        #This is the 4 motors... don't change.
        self.action_size = 4 
        
        self.phase = 0
        
        self.debug = debug

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        self.target_velocity = np.array([1., 1., 1.,]);

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        # TIny reward for just moving towards the target
        # This is kind of like "taking chess pieces"... free rewards... not great?
        #reward = 1.-.9*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
       
        
        #print(reward);
        
        #reward = 0
        #wrong_direction_discount = 1
        
        # print(self.sim.pose)
        #z_distance = abs(self.target_pos[2] - self.sim.pose[2])
        #distance = abs(self.sim.pose[:3] - self.target_pos)

        
        #if ((self.sim.v).sum() < 3):
        #    reward = reward * 0.5
    
        #sys.stdout.flush()
        #print('\r z position', self.sim.pose[2])#, 'z distance', z_distance, 'z target', self.target_pos[2])
        
        
        #if (self.target_pos[2] <= 0):
        #    wrong_direction_discount = 0.2
        
        # Discount for over target.
        #high_flyer_discount = 1
        #if (self.sim.pose[2] > (self.target_pos[2] + 2)):
        #    high_flyer_discount = 0.9
        #print(self.sim.pose[2])
        #if (self.sim.pose[2] > 0 and self.sim.pose[2] < self.target_pos[2] + 10):
        #Small reward for going up in general
        
            #reward = 0.1
            
            # DO NOT CHANGE THE REWARD FUNCTION!!! TRY TO REPLICATE THE PERFECT RESULT WITH THIS AGAIN
            # Happened once yesterday.. tweak the actor and the critic.
            
        # THE REWARD FUNCTION FOR ALL THE TESTS!!! 27/5/19
        #if (self.sim.pose[2] > 0):
            #reward = self.sim.pose[2] * 0.01
            
            
        #if (self.sim.pose[2] > 0):
        #    reward = self.sim.pose[2] * 0.01
            
        #if (self.sim.pose[2] > 1):
            #reward = 1 - 0.2*(1 / self.sim.pose[2])
            # Possible to make this a function that equals target? This currently makes
            # the copter hover at around 42... (where reward is very close to 1 per episode).
            # NOT VERY SPARSE...
            
        #if (self.sim.pose[2] > 11):
            #reward = 0

        # Large reward for being within range of the target z axis point, that increases the closer we get.
        #if (z_distance < 5):
            #reward = 100 / z_distance
        
        #print('distance');
        #print(distance.sum());
        # Todo: Double check distances are definitely absolute, getting high z values in training.
        # reward is also going up with higher z values for some reason.. should penalise?
        # Get largest target value for(x,y,z): get largest
        #      if largest > target. Penalise strictly.
        #reward = [distance[0] < 1, distance[1] < 1, distance[2] < 1].count(True)
        
        #if (self.sim.v < 0):
        #   reward 
        #if (reward == 1):
            #reward = 0.001;
        
        #if (reward == 2):
            #reward = 0.01;
            
        #if (reward == 3):
        #    reward = 1;
        #else:
        #    reward = 0
        
         # Euclidean distance. Source?
        #distance = abs(np.linalg.norm(self.sim.pose[:3]-self.target_pos))
        #reward = 1/distance
        #reward = -abs(np.linalg.norm(self.sim.pose[:3]-self.target_pos))
        #reward = -np.linalg.norm(self.sim.pose[:3]-self.target_pos)
        #reward = -np.linalg.norm(self.sim.pose[:3]-self.target_pos)
        #reward = np.clip(2.-.25*(np.linalg.norm(self.sim.pose[:3]-self.target_pos)), -1, 1)
        
        distance = abs(self.sim.pose[:3] - self.target_pos)
        
        #print('max', max(distance))
        
        # Emphasis penalty on further distances.
        # How to emphasis reward on better results?
        ec_dist = abs(np.linalg.norm(self.sim.pose[:3]-self.target_pos)) # THE GOOD RESULT
        #reward = (1-(np.power(max(distance), 2) / 100) + 2) * 2
        reward = (1-(np.power(ec_dist, 2) / 100) + 2) # THE GOOD RESULT
        #reward = 1-ec_dist
        #reward = 1-ec_dist
        
        
        
#        if (ec_dist < 5):
 #           reward
        
        #print(distance);
        #print(reward)
        #print('relu', (abs(reward) + reward))
        
        #print(np.linalg.norm(self.sim.pose[:3]-self.target_pos));
        #print(-np.tanh(reward))
        #print(distance.sum());
        #print(np.tanh(np.sinh(reward)));
        #print('----------')
        
        # End the episode if we reach the target.
        # Maybe remove this AFTER I can reach targets, then I can work on hovering.
        reached_target = ([distance[0] < 1, distance[1] < 1, distance[2] < 1].count(True) == 3);
        if (reached_target):
        #if (reward > -2):
            if (self.debug):
                print('REACHED TARGET THIS EPISODE')
            #print('distance', distance)
            #print('reward', reward)
            # THIS IS WORKING CORRECTY. VERIFIED IN TRAINER.
            self.sim.done = True
            # Makes sense.. no more "future rewards" for the agent to consider and abuse.
            return 100
        # Check if done is true before the runtime finished
        # crash penalty! https://github.com/WittmannF/quadcopter-best-practices
        elif (self.sim.done and self.sim.runtime > self.sim.time):
            if (self.debug):
                print('\nCRASHED')
            #print('distance', distance)
            #print('reward', reward)
            # THIS IS WORKING CORRECTY. VERIFIED IN TRAINER.
            #return -100
        
        # Give reward for not crashing?
        if ((self.sim.runtime - self.sim.time) < 0):
            if (reward < 0):
                reward = reward / 2
            if (reward > 0):
                reward = reward * 2
            #reward += 2
            if (self.debug):
                print('\nNO CRASH')
            #print(self.sim.runtime - self.sim.time)
            #reward = reward + 10
        
       
        
        #print('vel', np.mean(self.sim.v))
        #if (np.mean(self.sim.v > 0) and np.mean(self.sim.v < 1)):
        #   reward = reward + 2
        
        # ReLu activation for positive rewards.
        return (abs(reward) + reward) / 2 #np.tanh(reward)
        
            #velocity discount?
     
        
        #print(reward)
        #reward = 0.001*(reward**9)
        
        #velocity_discount = 1.-.9*(abs(self.sim.v - self.target_velocity)).sum()
        #print(velocity_discount)
        #print(self.sim.v)
        #reward = reward / velocity_discount
        #if (self.sim.v < 0):
        #    reward = 0
        
        #if (self.sim.v > 1):
        #    reward = 0
    
            
        #nice idea, but too much reward for nothing!! 2... Can I make it exponential?
                       
        #if (distance[0] < 1 and distance[1] < 1 and distance[2] < 1):
        #    reward = 1
        
        #if (distance[2] < 1):
            
            
        # let's try the x and y penalties.
        #if (self.sim.pose[0] < 1 or self.sim.pose[0] > -1):
            #reward = 1
            
        #if (self.sim.pose[1] < 1 or self.sim.pose[1] > -1):
            #reward = 1
       
        #WHY is the average reward still increasing past 15 height??
        # Ah... because rewards after about 10 get very close to 1!! Which are good rewards...
        # Maybe just give a flat reward?
        
        # tanh scales reward to -1 to 1.
        #return reward
   

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state