import tensorflow as tf
import os
from model import train, test
from configuration import get_config
import errno    
import os

config = get_config()
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

if __name__ == "__main__":
    # start training
    if config.train:
        print("\nTraining Session")
        mkdir_p(config.model_path)
        train(config.model_path)
    # start test
    else:
        print("\nTest session")
        if os.path.isdir(config.model_path):
            test(config.model_path)
        else:
            raise AssertionError("model path doesn't exist!")