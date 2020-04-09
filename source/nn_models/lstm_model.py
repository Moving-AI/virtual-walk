import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

logger = logging.getLogger(__name__)

class LSTMModel:
    def __init__(self, classes, input_dim, load_path_NN=None, lr=0.01, time_steps=5, optimizer='sgd', tensorboard_path=None):
        self.classes = classes
        self.n_classes = len(classes)
        self.time_steps = time_steps

        if load_path_NN is None:
            self.NN = self._get_NN(input_dim, self.n_classes)
        else:
            self.NN = self.load_NN(load_path_NN)
        self._compile_NN(optimizer, lr)

        if tensorboard_path is not None:
            self.callbacks = [self.create_callbacks(tensorboard_path)]
        else:
            self.callbacks = []

    def _get_NN(self, input_dim, output_dim):
        inputs = Input(shape=(self.time_steps, input_dim))
        x = LSTM(32, return_sequences=False)(inputs)
        outputs = Dense(output_dim, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def to_categorical(self, Y):
        if Y is not None:
            Y = np.array([self.classes.index(i) for i in Y])
            return tf.one_hot(Y, depth=self.n_classes)
        else:
            return None

    def train_NN(self, X_train, Y_train, batch_size, epochs, X_test=None, Y_test=None, is_one_hot=False, savepath=None,
                 verbose=0, callbacks=[]):
        '''
        Method to train the neural network.
        :param X_train, Y_train: ndarray. Training data. X contains the coordinates from PCA (shape (samples, n_components))
        and Y contains the labels (shape (samples,))
        :param batch_size: int.
        :param epochs: int.
        :param X_test, Y_test: ndarray.
        :param is_one_hot: bool. Format of the labels. These are converted to a one hot array in order to be fed to the NN.
        :param savepath: str.
        :param verbose: int.
        :return:
        '''
        if not is_one_hot:
            Y_train = self.to_categorical(Y_train)
            Y_test = self.to_categorical(Y_test)

        self.callbacks.extend(callbacks)

        history = self.NN.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=epochs,
                              callbacks=self.callbacks, verbose=verbose)

        if X_test is not None:
            assert Y_test is not None, 'In order to evaluate the model Y_test must be passed to the training function'
            self.NN.evaluate(X_test, Y_test)

        if savepath is not None:
            self.save_NN(savepath)

        return history

    def predict_NN(self, X, threshold_nn):
        Y = self.NN.predict(X)
        predicted_classes = np.argmax(Y, axis=1)
        return [self.classes[predicted_classes[i]] if Y[i, predicted_classes[i]] > threshold_nn else 'stand' for i in
                range(len(predicted_classes))], Y

    @staticmethod
    def load_NN(model_path):
        return tf.keras.models.load_model(model_path)

    def save_NN(self, savepath):
        self.NN.save(savepath)
        logging.debug('Neural network saved to ' + savepath)

    def _compile_NN(self, optimizer, lr):
        if optimizer == 'sgd':
            opt = SGD(lr=lr, decay=3e-5, momentum=0.9, nesterov=True)
        elif optimizer == 'adam':
            opt = Adam(learning_rate=lr)
        else:
            raise ValueError('Not implemented compiler')
        self.NN.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])


    @staticmethod
    def create_callbacks(path):
        return TensorBoard(log_dir=path, write_graph=True, histogram_freq=5, update_freq='epoch',
                           profile_batch=100000000,
                           write_grads=True)

    @staticmethod
    def prepare_input(data, time_steps):
        n_joints = 14
        Y = data[:, -1].astype('str')
        X = data[:, :-1].astype('f')
        X = X[:, :n_joints * 2 * time_steps]
        X = X.reshape((X.shape[0], time_steps, n_joints * 2))
        return X, Y