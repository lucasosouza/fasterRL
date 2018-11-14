# coding: utf-8

import gym
import numpy as np
import cv2

def make_env(env_name):
    """ Currently no transformations are required """
    
    env = gym.make(env_name)    
    return env


def wrap_env_malmo(env):

	env = ScaledFloatFrame(env)
	return env


def wrap_env_marlo(env):

    env = DownscaleFrame(env)
    env = ScaledFloatFrame(env)
    env = ImageToPyTorch(env)
    return env


class DownscaleFrame(gym.ObservationWrapper):
    """ Downscale image to expected size of 84x84 """ 

    def observation(self, obs):

        # last argument. can use intercubic - better but slower. default is interlinear

        return cv2.resize(obs, (84,84)) 

class ImageToPyTorch(gym.ObservationWrapper):

    """ Make transformation required for PyTorch: move channels to last dimension 

		No longer required since making modifiction directly to minecraft env library 
		to do frame stacking
    """
        
    def __init__(self, env):
        # init as all others
        super(ImageToPyTorch, self).__init__(env)
        # need to change observation space to reflect changes
        old_shape = self.observation_space.shape
        # troca o observation space
        #self.observation_space = gym.spaces.Box(low=0.0, high=1.0, 
        #    shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)
        # fixar o tamanho por hora
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, 
           shape=(old_shape[-1], 84, 84), dtype=np.float32)
        
    def observation(self, obs):
        """ Receives observation and returns observations with shifted axis """

        # move axis        
        return np.moveaxis(obs, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """ Converts observation data from integers to float
        and scales every pixel to range [0.0 ... 1.0]
        
        Why this method has no init, as others??
        Because it doees not require to change any attribute.
    """
    
    def observation(self, obs):

    	# transforma em float de 0 a 1
        return np.array(obs).astype(np.float32) / 255.