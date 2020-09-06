from fasterrl.common.wrapper import *
from fasterrl.common.discretizer import *
import numpy as np

# save expected reward and number of episodes
# will facilitate future reference
solved_requirements = {
    "MountainCarContinuous-v0": (90, 1),
    "CartPole-v0": (195,100),
}

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
            self.env = self.configure_gym_minecraft(params["ENV_NAME"])
            self.env = wrap_env_malmo(self.env)
            self.configure_gym()
        elif self.platform == "marlo":
            self.env = self.configure_gym_marlo(params["ENV_NAME"])
            self.env = wrap_env_malmo(self.env)
            self.configure_gym()

        self.render = False
        if "RENDER" in params:
            self.render = params["RENDER"]

        self.discretize_state = False

        # overrides default for experience sharing, setting for passive mode
        self.sharing = False
        if "SHARING" in params:
            self.sharing = params["SHARING"]

        self.focused_sharing = False
        if "FOCUSED_SHARING" in params:
            self.focused_sharing = params["FOCUSED_SHARING"]

        self.passive = False
        if self.sharing or self.focused_sharing:
            self.passive = True
            self.discretize_state = True

        # implements state discretization
        if "DISCRETIZE_STATE" in params:
            self.discretize_state = params["DISCRETIZE_STATE"]

        if self.discretize_state:
            # get bin size, if available
            bin_size = False
            if "DISCRETIZE_STATE_BIN_SIZE" in params:
                bin_size = params["DISCRETIZE_STATE_BIN_SIZE"]
            # or bin sizes for all vars, if availables
            bin_sizes = False
            if "DISCRETIZE_STATE_BIN_SIZES" in params:
                bin_sizes = params["DISCRETIZE_STATE_BIN_SIZES"]

            # see whether or not to use tile coding
            with_tiles = False
            if "WITH_TILES" in params:
                with_tiles = params["WITH_TILES"]

            tile_offsets = None
            if "TILE_OFFSETS" in params:
                tile_offsets = params["TILE_OFFSETS"]

            # initialize discretizer
            if with_tiles:
                self.state_discretizer = TileDiscretizer(self.observation_space, bin_size, bin_sizes, tile_offsets)
            else:
                self.state_discretizer = Discretizer(self.observation_space, bin_size, bin_sizes)

            discretize_type = 'unitary'
            if "DISCRETIZE_STATE_TYPE" in params:
                discretize_type = params["DISCRETIZE_STATE_TYPE"]
                # if not unitary, do sampling
                discretize_sampling_size = 1000
                if "DISCRETIZE_STATE_SAMPLING_SIZE" in params:
                    discretize_sampling_size = params["DISCRETIZE_STATE_SAMPLING_SIZE"]
                if discretize_type == 'true_sampling':
                    self.random_true_sampling(discretize_sampling_size, type='state')
                if discretize_type == 'false_sampling':
                    self.random_false_sampling(discretize_sampling_size, type='state')

            # change observation space
            if not self.passive:
                self.observation_space.shape = tuple(self.state_discretizer.bin_sizes)

        # implements action discretization
        self.discretize_action = False
        if "DISCRETIZE_ACTION" in params:
            self.discretize_action = params["DISCRETIZE_ACTION"]

        if self.discretize_action:
            # get bin size, if available
            bin_size = False
            if "DISCRETIZE_ACTION_BIN_SIZE" in params:
                bin_size = params["DISCRETIZE_ACTION_BIN_SIZE"]
            # or bin sizes for all vars, if availables
            bin_sizes = False
            if "DISCRETIZE_ACTION_BIN_SIZES" in params:
                bin_sizes = params["DISCRETIZE_ACTION_BIN_SIZES"]

            # initialize discretizer
            self.action_discretizer = ActionDiscretizer(self.action_space, bin_size, bin_sizes)

            discretize_type = 'unitary'
            if "DISCRETIZE_ACTION_TYPE" in params:
                discretize_type = params["DISCRETIZE_ACTION_TYPE"]
                # if not unitary, do sampling
                discretize_sampling_size = 1000
                if "DISCRETIZE_ACTION_SAMPLING_SIZE" in params:
                    discretize_sampling_size = params["DISCRETIZE_ACTION_SAMPLING_SIZE"]
                # no true sampling allowed for actions, only false sampling
                if discretize_type == 'sampling':
                    self.random_false_sampling(discretize_sampling_size, type='action')

            # change action space
            if not self.passive:
                self.action_space.shape = ()
                self.action_space.n = int(np.prod(self.action_discretizer.bin_sizes))

        # unpack relevant parameters
        self.reward_scaling_factor = None
        if "REWARD_SCALING_FACTOR" in params:
            self.reward_scaling_factor = params["REWARD_SCALING_FACTOR"]

        self.step_vars = {}
        self.episode_vars = {}

    # redirect several methods
    def step(self, action, sampling=False):

        # discretize action, if required
        if self.discretize_action and not self.passive:
            action = self.action_discretizer.revert(action)

        observation, reward, done, info = self.env.step(action)

        # short circuit and return before if stepping for samples only
        if sampling:
            return observation, reward, done, info

        # scale reward
        if self.reward_scaling_factor:
            reward *= self.reward_scaling_factor

        # do rendering, if required
        if self.platform == 'malmo':
            if self.render:
                self.env.render('human') # specific for minecraft

        # discretize observation, if required
        if self.discretize_state and not self.passive:
            observation = self.state_discretizer.convert(observation)

        return observation, reward, done, info

    def configure_gym(self):
        """ Redirect openai relevant functions """

        self.action_space  = self.env.action_space
        self.observation_space = self.env.observation_space

        # print("Observation Space: ", self.env.observation_space)
        # print("Action Space: ", self.env.action_space)


        # think later about adding random seed, pros vs cons
        # self.env.seed(random_seed)

    def configure_gym_marlo(self, env_name):
        """ Configure environment for Marlo platform
            Require previously launched minecraft environment
            - to launch: $MALMO_MINECRAFT_ROOT/launchClient.sh -p 10000

            Available params defined in:
            - https://github.com/crowdAI/marLo/blob/8652f8daef2caf9202881d002a2d3c28c882d941/marlo/base_env_builder.py
        """

        import marlo
        marlo.logger.setLevel('ERROR')

        # client_pool = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
        client_pool = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
        join_tokens = marlo.make(env_name,
                                 params={
                                    "client_pool": client_pool,
                                    "videoResolution" : [84,84],
                                    "tick_length": 1,
                                    # "prioritise_offscreen_rendering": False,
                                    # "comp_all_commands": ['move', "turn"],
                                 })
        join_token = join_tokens[0]
        env = marlo.init(join_token)

        return env

    def configure_gym_minecraft(self, env_name):
        """ Configure environment for gym minecraft platform """

        import gym
        import gym.spaces
        import gym_minecraft

        env = gym.make(env_name)
        env.configure(client_pool=[('127.0.0.1', 10000), ('127.0.0.1', 10001)])
        env.configure(allowDiscreteMovement=["move", "turn"]) # , log_level="INFO")
        env.configure(videoResolution=[84,84])
        env.seed(42)

        return env

    def reset(self):
        observation = self.env.reset()

        # discretize observation, if required
        if self.discretize_state and not self.passive:
            observation = self.state_discretizer.convert(observation)

        return observation

    def report_step(self):
        return self.step_vars

    def report_episode(self):
        return self.episode_vars

    def random_false_sampling(self, count, type='state'):

        samples = []

        # randomly samples states
        if type == 'state':
            for i in range(count):
                samples.append(self.env.observation_space.sample())

            self.state_discretizer.define_bins_from_samples(samples)

        # randomly sample actions
        elif type == 'action':
            for i in range(count):
                samples.append(self.env.action_space.sample())

            self.action_discretizer.define_bins_from_samples(samples)


    def random_true_sampling(self, count, type='state'):

        samples = []
        # interact with environment untill it gets all samples
        self.reset()
        for i in range(count):
            obs, _, done, _ = self.step(self.env.action_space.sample(), sampling=True)
            samples.append(obs)
            if done:
                self.reset()
        # reset after finishing sampling
        self.env.reset()

        self.state_discretizer.define_bins_from_samples(samples)



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



