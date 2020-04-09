import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.decomposition import PCA
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import SGD, Adam

logger = logging.getLogger(__name__)


class FullModel:  # Not Model because it would shadow keras'
    def __init__(self, classes, load_path_scaler=None, load_path_PCA=None, load_path_NN=None, n_components=50,
                 layers_NN=[50, 50], lr=0.01, dropout=0, optimizer='sgd', tensorboard_path=None):
        '''
        This model consists of a PCA and Neural Network. It has all the necessary methods to train and predict all the
        results.
        FLOW:
            1. Read data in the form
            [['stand', '1', '2', '3'],
             ['walk', '4', '5', '6']]

            2. Create model = FullModel()
            3. model.prepare_x_y(data)
            4. model.train_PCA(X)
            5. X_pca = model.predict_PCA(X)
            6. model.train_NN(X_pca, Y)
            7. model.predict(X_test)
        :param classes: list. Different classes the neural network has to be trained in. ['stand', 'walk']?
        :param load_path_PCA: str. If the PCA model has to be loaded from a location, the path to that location
        :param load_path_NN: str. If the Neural Network has to be loaded from a location, the path to that location
        :param n_components: int. The number of components the PCA has to consider. Also, the input dimension for the
        Neural Network
        :param layers_NN: list. This list specifies the architecture of the neural network. If [50, 50], two dense layers
        with 50 neurons each will be created. Only used if load_path_NN is None.
        :param lr, decay, momentum: float. Learning rate, decay and momentum for the SGD optimizer in the neural network
        :param tensorboard_path: str. The path to where the tensorboard logs will be saved. If set to none, no callbacks
        will be used.
        '''
        self.classes = classes
        self.n_classes = len(classes)
        if load_path_scaler is None:
            self.scaler = preprocessing.StandardScaler()
        else:
            self.scaler = self.load_model(load_path_scaler)

        if load_path_PCA is None:
            self.PCA = PCA(n_components)
        else:
            self.PCA = self.load_model(load_path_PCA)

        if load_path_NN is None:
            self.NN = self._get_NN(n_components, self.n_classes, layers_NN, dropout)
        else:
            self.NN = self.load_NN(load_path_NN)
        self._compile_NN(optimizer, lr)

        if tensorboard_path is not None:
            self.callbacks = [self.create_callbacks(tensorboard_path)]
        else:
            self.callbacks = []

    def predict(self, X, threshold_nn):
        '''
        Function that predicts the class when data X is given. A threshold is applied if specified.
        Args:
            X: ndarray. data
            threshold_nn: float. Threshold to apply to the results. If the probability of the most probable action does not
            exceed the threshold, "stand" is returned

        Returns:
            Predicted class (list) and probabilities for each class (list)
        '''
        X_scaler = self.predict_scaler(X)
        X_trans = self.predict_PCA(X_scaler)
        predicted_class, probs = self.predict_NN(X_trans, threshold_nn)
        return predicted_class, probs

    def train_scaler(self, X, savepath=None):
        self.scaler.fit(X)
        if savepath is not None:
            self.save_scaler(savepath)

    def predict_scaler(self, X):
        return self.scaler.transform(X)

    def train_PCA(self, X, savepath=None):
        self.PCA.fit(X)
        if savepath is not None:
            self.save_PCA(savepath)

    def to_categorical(self, Y):
        if Y is not None:
            Y = np.array([self.classes.index(i) for i in Y])
            return tf.one_hot(Y, depth=self.n_classes)
        else:
            return None

    def _compile_NN(self, optimizer, lr):
        if optimizer == 'sgd':
            opt = SGD(lr=lr, decay=3e-5, momentum=0.9, nesterov=True, clipnorm=1.)
        elif optimizer == 'adam':
            opt = Adam(learning_rate=lr, clipnorm=0.5)
        else:
            raise ValueError('Not implemented compiler')
        self.NN.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

    def train(self, X_train, Y_train, batch_size, epochs, X_test=None, Y_test=None, is_one_hot=False, callbacks=[], verbose=0):
        self.train_scaler(X_train)
        X_scaled = self.predict_scaler(X_train)
        X_scaled_test = self.predict_scaler(X_test)

        self.train_PCA(X_scaled)
        logging.info('Explained variance: {}'.format(self.get_explained_variance_ratio()))
        X_pca = self.predict_PCA(X_scaled)
        X_pca_test = self.predict_PCA(X_scaled_test)

        history = self.train_NN(X_pca, Y_train, X_test=X_pca_test, Y_test=Y_test, batch_size=batch_size, epochs=epochs,
                      is_one_hot=is_one_hot, callbacks=callbacks, verbose=verbose)
        return history

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

    def predict_PCA(self, X):
        return self.PCA.transform(X)

    def predict_NN(self, X, threshold_nn):
        Y = self.NN.predict(X)
        predicted_classes = np.argmax(Y, axis=1)
        return [self.classes[predicted_classes[i]] if Y[i, predicted_classes[i]] > threshold_nn else 'stand' for i in
                range(len(predicted_classes))], Y

    def _get_NN(self, input_dim, output_dim, layers, dropout):
        '''
        Creation of the neural network.
        :param input_dim: integer.
        :param output_dim: integer.
        :param layers: list. This list specifies the architecture of the neural network. If [50, 50], two dense layers
        with 50 neurons each will be created.
        :return: neural network model.
        '''
        inputs = Input(shape=(input_dim,))
        x = self._Dense(inputs, layers[0], dropout, 'relu')
        for layer in layers[1:]:
            x = self._Dense(x, layer, dropout, 'relu')
        outputs = Dense(output_dim, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def _Dense(x, dim, dropout, activation='relu'):
        x = Dense(dim, activation=activation, kernel_initializer='he_normal', bias_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01),
                  activity_regularizer=tf.keras.regularizers.l2(0.01))(x)
        #
        if dropout != 0:
            x = Dropout(dropout)(x)
        return x

    def save_scaler(self, savepath):
        if savepath is None:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            savepath = Path(__file__).parents[1].joinpath('models/scaler_{}.pkl'.format(current_time))

        with open(savepath, 'wb') as file:
            pickle.dump(self.scaler, file)
        logging.debug('Scaler model saved to ' + savepath)

    def save_PCA(self, savepath):
        if savepath is None:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            savepath = Path(__file__).parents[1].joinpath('models/PCA_{}.pkl'.format(current_time))

        with open(savepath, 'wb') as file:
            pickle.dump(self.PCA, file)
        logging.debug('PCA model saved to ' + savepath)

    @staticmethod
    def load_model(path_model):
        with open(path_model, 'rb') as file:
            pickle_model = pickle.load(file)

        return pickle_model

    @staticmethod
    def load_NN(model_path):
        return tf.keras.models.load_model(model_path)

    def save_NN(self, savepath):
        self.NN.save(savepath)
        logging.debug('Neural network saved to ' + savepath)

    @staticmethod
    def prepare_x_y(data):
        Y = data[:, -1].astype('str')
        X = data[:, :-1].astype('f')
        return X, Y

    @staticmethod
    def create_callbacks(path):
        return TensorBoard(log_dir=path, write_graph=True, histogram_freq=5, update_freq='epoch',
                           profile_batch=100000000,
                           write_grads=True)

    def get_explained_variance_ratio(self):
        return sum(self.PCA.explained_variance_ratio_)
