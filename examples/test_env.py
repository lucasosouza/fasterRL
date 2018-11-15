import sys
sys.path.append("../../")

from fasterRL.common.environment import *

params = {}
params["PLATFORM"] = "openai"
params["ENV_NAME"] = "MountainCarContinuous-v0"

env = BaseEnv(params)
import pdb;pdb.set_trace()

