import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from omegaconf import OmegaConf
from floral.utils import eval_num
if not OmegaConf.has_resolver("eval_num"):
    OmegaConf.register_new_resolver("eval_num", eval_num)
from .floral import Floral
# maybe catch a ModuleNotFoundError and remind user to activate conda as a nice touch