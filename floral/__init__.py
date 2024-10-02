
def _meta_init():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # warnings.filterwarnings("ignore", category=FutureWarning)
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from omegaconf import OmegaConf
    from floral.utils import eval_num
    if not OmegaConf.has_resolver("eval_num"):
        OmegaConf.register_new_resolver("eval_num", eval_num)
    try:
        # Checking for cuda is memory-intensive, which is exacerbated in a large actor pool,
        # so we turn off GPUs for TensorFlow as it is only used for data loading.
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    except ModuleNotFoundError:
        pass


_meta_init()
from .floral import Floral
