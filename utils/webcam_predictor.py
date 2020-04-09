import logging
import time
from copy import deepcopy
from pathlib import Path

import cv2

from utils.controller import Controller
from utils.dataprocessor import DataProcessor
from utils.lstm_model import LSTMModel
from utils.model import FullModel
from utils.person_frames import PersonMovement

FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)

formatter = logging.Formatter(FORMAT)
logger.setLevel(logging.INFO)


class WebcamPredictor:
    def __init__(self, classes=["walk", "stand", "left", "right"], model='LSTM', pca_model_path=None,
                 nn_model_path=None, pose_model_path=None, scaler_model_path=None, output_video_dim=(640, 480),
                 coordinates=None, default_limit=None, driver_path=None, threshold_nn=0.5, time_rotation=0.5):

        self.n_frames = 5
        self.threshold_nn = threshold_nn
        self.classes = classes
        self.model = model
        logging.info('Using {} model'.format(model))

        if self.model == 'LSTM':
            if nn_model_path is not None:
                LSTM_PATH = Path(nn_model_path)
            else:
                LSTM_PATH = Path(__file__).parents[1].joinpath("models/LSTM.h5")

            self.model_lstm = LSTMModel(
                classes,
                input_dim=28,
                load_path_NN=str(LSTM_PATH)
            )
            self.process_frames = self.process_list_lstm
        else:
            if pca_model_path is not None:
                PCA_PATH = Path(pca_model_path)
            else:
                PCA_PATH = Path(__file__).parents[1].joinpath("models/PCA.pkl")

            if nn_model_path is not None:
                NN_PATH = Path(nn_model_path)
            else:
                NN_PATH = Path(__file__).parents[1].joinpath("models/NN.h5")

            if scaler_model_path is not None:
                SCALER_PATH = Path(scaler_model_path)
            else:
                SCALER_PATH = Path(__file__).parents[1].joinpath("models/SCALER.pkl")

            self.model = FullModel(
                classes=self.classes,
                load_path_PCA=str(PCA_PATH),
                load_path_NN=str(NN_PATH),
                load_path_scaler=str(SCALER_PATH)
            )
            self.process_frames = self.process_list

        if pose_model_path is None:
            POSE_PATH = Path(__file__).parents[1].joinpath(
                "models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")
        else:
            POSE_PATH = pose_model_path
        rescale = output_video_dim[0] / 257, output_video_dim[1] /257
        self.processor = DataProcessor(POSE_PATH, rescale=rescale)

        if coordinates is not None:
            self.controller = Controller(self.classes, coordinates=coordinates, driver_path=driver_path,
                                         time_rotation=time_rotation)
        else:
            self.controller = Controller(self.classes, time_rotation=time_rotation)

        if default_limit is None:
            default_limit = 0.5

        initial_time = time.time()
        self.last_calls = {element: [initial_time, default_limit] for element in self.classes}

        self.font, self.color = self._prepare_painter()

    def _prepare_painter(self):
        font = cv2.FONT_HERSHEY_PLAIN
        color = (131, 255, 51)
        return font, color

    def predictor(self, output_dim=None, show_skeleton=False, times_v=1):
        probabilities = None
        network_frame_size = (257, 257)
        capture = cv2.VideoCapture(0)
        if output_dim is None:
            output_dim = (int(capture.get(4)), int(capture.get(3)))

        buffer = []
        buffer_og = []  # For populating future buffers
        valid = 0
        while True:
            _, frame_orig = capture.read()
            frame = cv2.resize(frame_orig, network_frame_size, interpolation=cv2.INTER_LINEAR)

            person = self.processor.process_live_frame(frame)

            if valid == 0 and person.is_valid_first():
                frame = cv2.resize(frame, output_dim[::-1], interpolation=cv2.INTER_LINEAR)
                # cv2.imshow("WebCam", frame)
                buffer.append(person)
                buffer_og.append(person)
                valid += 1

            elif 0 < valid < self.n_frames - 1 and person.is_valid_other():
                # If valid as first, take into account for future frames
                if person.is_valid_first():
                    buffer_og.append(deepcopy(person))
                else:
                    buffer_og.append(False)

                person.infer_lc_keypoints(buffer[valid - 1])

                buffer.append(person)
                valid += 1
            elif valid == self.n_frames - 1 and person.is_valid_other():
                # Here is the ONLY case in which we process a group of frames
                # If frame was valid for first initially, take into account for future frames
                if person.is_valid_first():
                    buffer_og.append(deepcopy(person))
                else:
                    buffer_og.append(False)
                person.infer_lc_keypoints(buffer[valid - 1])

                buffer.append(person)

                probabilities = self.process_frames(buffer, times_v)

                valid_startings = [i for i, person in enumerate(buffer_og) if person != False]
                if len(valid_startings) > 0:
                    buffer = buffer_og[valid_startings[0]:]
                    valid = len(buffer)
                else:
                    buffer = []
                    valid = 0
            elif person.is_valid_first():
                buffer = [person]
                buffer_og = [person]
                valid = 1

            else:
                buffer = []
                valid = 0

            if show_skeleton and probabilities is not None:
                person.draw_points(frame_orig)
                self._write_probabilities(frame_orig, probabilities)
                self._write_distance(frame_orig, self.controller.distance_calculator.distance)
                cv2.imshow('frame', frame_orig)

            # End of while
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def process_list(self, buffer, times_v):
        person_movement = PersonMovement(buffer, times_v, model=self.model)

        prediction, probabilities = self.model.predict(person_movement.coords, self.threshold_nn)
        prediction = prediction[0]
        probabilities = probabilities[0]
        if time.time() - self.last_calls[prediction][0] > self.last_calls[prediction][1]:
            self.last_calls[prediction][0] = time.time()
            self.controller.perform_action_name(prediction)

        return probabilities

    def process_list_lstm(self, buffer, *args):
        '''
        Processs a list of frames with LSTM.
        Args:
            buffer:

        Returns:

        '''
        person_movement = PersonMovement(buffer, model='LSTM').coords
        prediction, probabilities = self.model_lstm.predict_NN(person_movement, self.threshold_nn)
        prediction = prediction[0]
        probabilities = probabilities[0]
        if time.time() - self.last_calls[prediction][0] > self.last_calls[prediction][1]:
            self.last_calls[prediction][0] = time.time()
            self.controller.perform_action_name(prediction)

        return probabilities

    def _write_probabilities(self, frame, probabilities):
        font = cv2.FONT_HERSHEY_PLAIN
        color = (131, 255, 51)
        for i, (p, c) in enumerate(zip(probabilities, self.classes)):
            pos = (10, 20 * (i + 1) + 50)
            cv2.putText(frame, '{}: {:.3f}'.format(c, p), pos, font, 0.8, color, 1)
        return frame

    def _write_distance(self, frame, distance):
        font = cv2.FONT_HERSHEY_PLAIN
        color = (131, 255, 51)
        # pos = (10, 20 * (4 + 1) + 50)
        pos = (10, 400)
        cv2.putText(frame, 'Distance: {:.3f}'.format(distance), pos, font, 1, color, 1)
        return frame
