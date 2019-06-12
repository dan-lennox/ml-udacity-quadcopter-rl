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

        # Min value of each action dimension
        self.action_low = 0
        # Max value of each action dimension
        self.action_high = 900

        #This is the 4 motors... don't change.
        self.action_size = 4 
        
        # Set debug flag.
        self.debug = debug

        # Goal (Fly to a target position).
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
       
        # Record the distance vector between the current copter position and the target position.
        distance = abs(self.sim.pose[:3] - self.target_pos)
         
        # Record the euclidean distance between the current copter position and the target position.
        ec_dist = abs(np.linalg.norm(self.sim.pose[:3]-self.target_pos))

        # Apply a small reward for progress towards the target.
        # Reward scales to penalise further distances heavily.
        reward = (1-(np.power(ec_dist, 2) / 100) + 2)
        
        # Record if the coptor came within one unit of the target in all dimensions. This is
        # counted as "reaching the target".
        reached_target = ([distance[0] < 1, distance[1] < 1, distance[2] < 1].count(True) == 3);

        if (reached_target):
            if (self.debug):
                print('REACHED TARGET THIS EPISODE')

            # End this episode early if the target was reached.
            self.sim.done = True
            
            # Large reward for actually reaching the target.
            return 100
        
        # Check to see if the copter crashed.
        elif (self.sim.done and self.sim.runtime > self.sim.time):
            if (self.debug):
                print('\nCRASHED')
            # Penalise heavily for crashes.
            # (Removed as performance worked better without this negative reward, see below for
            # the positive NOT crashing reward which replaced this behaviour.
            #return -100
        
        # If the copter is still in the air this timestep and hasn't crashed!
        if ((self.sim.runtime - self.sim.time) < 0):
            # Double the reward given based on the distance to the target.
            if (reward < 0):
                reward = reward / 2
            if (reward > 0):
                reward = reward * 2

            if (self.debug):
                print('\nNO CRASH')
        
        # ReLu activation for positive rewards.
        return (abs(reward) + reward) / 2

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