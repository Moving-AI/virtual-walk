import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp

from utils.model import FullModel

FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
formatter = logging.Formatter(FORMAT)
logger.setLevel(logging.INFO)

classes=['walk', 'stand', 'left', 'right']
LR = 0.001
components = 50
decay = 1e-6
momentum = 0.9
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
tb_path = Path(__file__).parents[0].joinpath('logs/{}/'.format(current_time))

data = np.loadtxt('data/training_data.txt', delimiter=',', dtype=object)
X, Y = FullModel.prepare_x_y(data)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# model = FullModel(classes, tensorboard_path=tb_path, n_components=components, lr=LR, decay=momentum)
# model.train(X_train, Y_train, X_test=X_test, Y_test=Y_test, batch_size=50, epochs=100)


HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([25, 50, 75, 100]))
LR = [x / 1000 for x in range(1, 100, 1)]
HP_LR = hp.HParam('learning rate', hp.Discrete(LR))
METRIC_ACCURACY = 'accuracy'

tb_path_0 = 'logs/' + current_time

session_num = 0
for batch_size in HP_BATCH_SIZE.domain.values:
    for lr in HP_LR.domain.values:
        hparams = {
            HP_BATCH_SIZE: batch_size,
            HP_LR: lr
        }
        tb_path = tb_path_0 + '_' + str(session_num)
        model = FullModel(classes, tensorboard_path=tb_path, lr=lr)
        model.train(X_train, Y_train, X_test=X_test, Y_test=Y_test, batch_size=batch_size, epochs=100, callbacks=[hp.KerasCallback(tb_path, hparams)])
        session_num += 1