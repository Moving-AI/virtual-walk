import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from utils.model import FullModel

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
formatter = logging.Formatter(FORMAT)
logger.setLevel(logging.INFO)

classes = ['walk', 'stand', 'left', 'right']

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
tb_path = Path(__file__).parents[0].joinpath('logs/{}/'.format(current_time))

data = np.loadtxt('data/training_data.txt', delimiter=',', dtype=object)
X, Y = FullModel.prepare_x_y(data)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

layers_struct = np.array([[50, 50], [50, 25], [25, 25]])

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([25, 50, 100]))
HP_LAYERS = hp.HParam('layers', hp.Discrete(list(range(len(layers_struct)))))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0., 0.2))
HP_LR = hp.HParam('learning rate', hp.RealInterval(0.0005, 0.005))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
METRIC_ACCURACY = 'accuracy'

tb_path_0 = 'logs/' + current_time

session_num = 0
for batch_size in HP_BATCH_SIZE.domain.values:
    for i_layer_struct in HP_LAYERS.domain.values:
        for dropout in tf.linspace(HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value, 3):
            for lr in tf.linspace(HP_LR.domain.min_value, HP_LR.domain.max_value, 20):
                for optimizer in HP_OPTIMIZER.domain.values:
                    layer_struct = layers_struct[i_layer_struct]
                    hparams = {
                        HP_BATCH_SIZE: batch_size,
                        HP_LR: f'{lr.numpy():.4f}',
                        HP_DROPOUT: f'{dropout.numpy():.1f}',
                        HP_LAYERS: i_layer_struct,
                        HP_OPTIMIZER: optimizer
                    }
                    tb_path = tb_path_0 + '_' + str(session_num) + '/'
                    model = FullModel(classes, tensorboard_path=tb_path, lr=lr.numpy(), n_components=50, layers_NN=layer_struct,
                                      dropout=dropout.numpy(), optimizer=optimizer)
                    model.train(X_train, Y_train, X_test=X_test, Y_test=Y_test, batch_size=batch_size, epochs=50,
                                callbacks=[hp.KerasCallback(tb_path, hparams)])
                    session_num += 1
