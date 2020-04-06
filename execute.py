import argparse

from utils.funciones import str2bool
from utils.webcam_predictor import WebcamPredictor

parser = argparse.ArgumentParser()

zaragoza = [41.6423746, -0.8862972]
paris = [48.8574788, 2.3515892]
roma = [41.9045284, 12.4941586]
barcelona = [41.3921363, 2.1618873]
nyc = [40.7576611, -73.9866615]
sidney = [-33.8643791, 151.2127931]
nueva_zelanda = [-41.2846378, 174.7772473]

parser.add_argument('-d', '--driver-path',
                    default='geckodriver.exe',
                    type=str,
                    help='The path to the Firefox driver')

parser.add_argument('-s', '--scaler-model',
                    default='models/scaler_0.pkl',
                    type=str,
                    help='Path to the scaler model.')

parser.add_argument('-p', '--pca-model',
                    default='models/pca_0.pkl',
                    type=str,
                    help='Path to the PCA model.')

parser.add_argument('-nn', '--nn-model',
                    default='models/NN_0.h5',
                    type=str,
                    help='Path to the neural network.')

parser.add_argument('-threshold',
                    default=0,
                    type=float,
                    help='Threshold for the neural network.')

parser.add_argument('-limit', '--default-limit',
                    default=0.5,
                    type=float,
                    help='Time to wait between consecutive actions.')

parser.add_argument('-r', '--time-rotating',
                    default=0.5,
                    type=float,
                    help='Time rotating the camera (in seconds) when detecting right or left actions.')

parser.add_argument('-c', '--initial-coordinates',
                    nargs='+',
                    type=float,
                    default=zaragoza,
                    help='Initial coordinates.')

parser.add_argument('-skeleton', type=str2bool, nargs='?',
                    const=True, default=True,
                    help="Show skeleton")

FLAGS, unparsed = parser.parse_known_args()

wp = WebcamPredictor(driver_path=FLAGS.driver_path, scaler_model_path=FLAGS.scaler_model,
                     pca_model_path=FLAGS.pca_model, nn_model_path=FLAGS.nn_model,
                     threshold_nn=FLAGS.threshold, default_limit=FLAGS.default_limit, time_rotation=FLAGS.time_rotating,
                     coordinates=FLAGS.initial_coordinates)

wp.predictor(times_v=1, show_skeleton=FLAGS.skeleton)
