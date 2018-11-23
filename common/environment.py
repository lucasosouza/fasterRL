from fasterRL.common.wrapper import *

class BaseEnv():
    """    
    make a class that overwrites the default environment
    and can encapsulate several things here
    environment step
    here I can consider the diferent environments I can use depending on the platform 
    """

    def __init__(self, params):

        # initialize environment depending on the platform
        if "PLATFORM" not in params:
            raise Exception("Please define the paramater PLATFORM. Currently supported plataforms: openai, marlo, gym-minecraft")

        self.platform = "openai"
        if "PLATFORM" in params:
            self.platform = params["PLATFORM"]

        self.render = False
        if "RENDER" in params:
            self.render = params["RENDER"]

        # some code reuse here can be done 
        if self.platform == "openai":
            import gym
            self.env = gym.make(params["ENV_NAME"])
            self.configure_gym()
        elif self.platform == "openai-atari":
            import gym
            self.env = gym.make(params["ENV_NAME"])
            self.env = wrap_env_atari(self.env)
            self.configure_gym()
        elif self.platform == "malmo":
            import gym
            import gym.spaces            
            import gym_minecraft
            self.env = gym.make(params["ENV_NAME"])
            self.configure_gym_minecraft()
            self.env = wrap_env_malmo(self.env)
            self.configure_gym()

        self.step_vars = {}
        self.episode_vars = {}

        # unpack relevant parameters
        self.reward_scaling_factor = None
        if "REWARD_SCALING_FACTOR" in params:
            self.reward_scaling_factor = params["REWARD_SCALING_FACTOR"]

    # redirect several methods
    def step(self, action):
        # what do I usually get when I step? 
        observation, reward, done, info = self.env.step(action)

        # scale reward
        if self.reward_scaling_factor:
            reward *= self.reward_scaling_factor

        # do rendering, if required
        if self.platform == 'malmo':
            if self.render:
                self.env.render('human') # specific for minecraft        

        return observation, reward, done, info

    def configure_gym(self):
        """ Redirect openai relevant functions """

        self.action_space  = self.env.action_space
        self.observation_space = self.env.observation_space

        # print("Observation Space: ", self.env.observation_space)
        # print("Action Space: ", self.env.action_space)


        # think later about adding random seed, pros vs cons
        # self.env.seed(random_seed)

    def configure_gym_minecraft(self):

        self.env.configure(client_pool=[('127.0.0.1', 10000), ('127.0.0.1', 10001)])
        self.env.configure(allowDiscreteMovement=["move", "turn"]) # , log_level="INFO")
        self.env.configure(videoResolution=[84,84])
        self.env.seed(42)

    def reset(self):
        state = self.env.reset()

        return state

    def report_step(self):
        return self.step_vars

    def report_episode(self):
        return self.episode_vars


#################

# register additional environments

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)



