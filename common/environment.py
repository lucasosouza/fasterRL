
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

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

        self.platform = params["PLATFORM"]

        if self.platform == "openai":
            import gym
            self.env = gym.make(params["ENV_NAME"])
            self.configure_openai()

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
        if self.reward_scaling_factor:
            reward *= self.reward_scaling_factor
        return observation, reward, done, info

    def configure_openai(self):
        """ Redirect openai relevant functions """

        self.action_space  = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        state = self.env.reset()

        return state

    def report_step(self):
        return self.step_vars

    def report_episode(self):
        return self.episode_vars

