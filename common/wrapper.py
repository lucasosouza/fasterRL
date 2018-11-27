# coding: utf-8

import gym
import numpy as np
import cv2

def make_env(env_name):
    """ Currently no transformations are required """
    
    env = gym.make(env_name)    
    return env

# test version without wrapper
def wrap_env_malmo(env):
    """ Can later define number of frames to keep in buffer as a parameter """ 

    env = BlackAndWhite(env)
    env = Downscale(env, 84, 84)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)

    return env


def wrap_env_marlo(env):

    env = Downscale(env)
    env = ScaledFloatFrame(env)
    env = ImageToPyTorch(env)
    return env

def wrap_env_atari(env):

    env = Downscale(env)
    env = ScaledFloatFrame(env)
    env = ImageToPyTorch(env)
    return env

class BlackAndWhite(gym.ObservationWrapper):
    """ Converts to black and white """ 

    def __init__(self, env):
        # init as all others
        super(BlackAndWhite, self).__init__(env)

        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, 
           shape=(old_shape[0], old_shape[1], 1), dtype=np.float32)

    def observation(self, obs):
#        print("original", obs.shape)
        obs = obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114
#        print("b&w", obs.shape)
        return obs

class Downscale(gym.ObservationWrapper):
    """ Downscale image to expected size of 84x84 """ 

    def __init__(self, env, height, width):
        # init as all others
        super(Downscale, self).__init__(env)

        self.resized_height = height
        self.resized_width = width

        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, 
           shape=(height, width, old_shape[-1]), dtype=np.float32)

    def observation(self, obs):
        # last argument. can use intercubic - better but slower. default is interlinear
        obs = cv2.resize(obs, (self.resized_height, self.resized_width)) 
        obs = np.expand_dims(obs, axis=-1) # hack, can fix
#        print("downscale", obs.shape)
        return obs

class FloatToInt(gym.ObservationWrapper):
    """ Convert from float to unsigned integer to reduce storage needs """ 

    def observation(self, obs):
        return x.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):

    # """ Make transformation required for PyTorch: move channels to last dimension 
    # """
        
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)

        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, 
           shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)
        
    def observation(self, obs):
        """ Receives observation and returns observations with shifted axis """

#        import pdb;pdb.set_trace()
        return np.moveaxis(obs, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """ Converts observation data from integers to float
        and scales every pixel to range [0.0 ... 1.0]        
    """
    
    def observation(self, obs):

        # transforma em float de 0 a 1
        return np.array(obs).astype(np.float32) / 255.

## Based on OpenAI Baselines

class FireResetEnv(gym.Wrapper):
        """ Presses the FIRE button in environments that require them in order for the game to start 
            Also check for corner cases present in some games
        """
        
        def __init__(self, env=None):            
            super(FireResetEnv, self).__init__(env) 

            # verify if action 1 meaning is fire
            assert env.unwrapped.get_action_meanings()[1] == 'FIRE'

            # verify if there are 3 or more possible actions
            assert len(env.unwrapped.get_action_meanings()) >= 3
            
        # def step(self, action):            
        #     return self.env.step(action)
    
        def reset(self):
            """ Overrides reset by implementing and additional functionality after reset
                Take action 1, if done reset again
                Take action 2, and if done, reset again
            """
            
            self.env.reset()
            obs, _, done, _ = self.env.step(1)
            if done: 
                self.env.reset()
            obs, _, done, _ = self.env.step(2)
            if done: 
                self.env.reset()
                
            return obs

class MaxAndSkipEnv(gym.Wrapper):
    """ Return only every 'skip'-th frame (accumulates reward between frames skipped)
        Max pool over X frames, to avoid flickering issues in Atari (defaults to 2)
    """    
    
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)

        # store most recent raw observations for max pooling across time
        self._obs_buffer = collections.deque(maxlen=2)

        self._skip = skip
        
    def step(self, action):
        """ Overrides step action
        
            Loop stores all observations during skip but will only keep the last X observations, 
            where X is the size of the obs_buffer
        
            Accumulates reward for all frames that skips
        """
        
        total_reward = 0.0
        done = None

        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done: break
        
        # take the max of the frames stored in buffer
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        
        return max_frame, total_reward, done, info
    
    def reset(self):
        """ Added implementation.
            Clear past frame buffer and init to first observation from inner environmet """
        
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        
        # redefines observation space to 84x84x1 frames, using 8bit unsigned int
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84,84,1), dtype=np.uint8
        )
        
    def observation(self, obs):
        """ Overrides observation method of environment """

        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):

        # converts into a 3d matrix and to float32
        if frame.size == 210*160*3:
            img = np.reshape(frame, [210,160,3]).astype(np.float32)
        elif frame.size == 250*160*3:
            img = np.reshape(frame, [250,160,3]).astype(np.float32)
        else:
            assert False, "Unknown resolution"

        # convert to grayscale
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

        # resize to 84x110, maintain proportions
        resized_screen = cv2.resize(img, (84,110), interpolation=cv2.INTER_AREA)

        # crop bottom and top of the image to make it square
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84,84,1])

        # convert to 8=bit unsigned int before returning
        return x_t.astype(np.uint8)

class BufferWrapper(gym.ObservationWrapper):
    """ 
        Stacks frame along first dimensions
        Goal is to return observation as a stack of last 4 frames, to give the agent an idea of the dynamics of the objects
        
        It is a smart hack, to solve the problem that RL problems deal with MDP (by definition, in a markovian process the current state holds all necessary information to derive future states)

        Unnecessary if using a function approximation method that keeps memoery, such as RNNs

    """
    
    def __init__(self, env, n_steps, dtype=np.float32):
        """ 
            Two additional arguments:
            - n_steps: number of frames to stack
            - dtype: data type to be used
        """        
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype

        # redefinition of observation space
        old_space = env.observation_space

        # repeat method?
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0), old_space.high.repeat(n_steps, axis=0),
            dtype=dtype)
        
    def reset(self):
        """ Overwrites reset to include initialization of buffer"""
        
        # start with all zeros
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)

        return self.observation(self.env.reset())
    
    def observation(self, observation):
        """ Add new observation to the bottom of the queue of observations 
        """
        
        # push backward all observations
        self.buffer[:-1] = self.buffer[1:]

        # add last observation to the end (similar to adding an item to a queue)
        self.buffer[-1] = observation
        
        return self.buffer