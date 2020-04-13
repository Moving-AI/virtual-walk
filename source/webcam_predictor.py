import logging
import time
from copy import deepcopy
from pathlib import Path
import yaml

import cv2

from source.controller import Controller
from source.dataprocessing import DataProcessor
from source.entities.person_frames import PersonMovement
from source.nn_models.lstm_model import LSTMModel
from source.nn_models.model import FullModel

FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

formatter = logging.Formatter(FORMAT)
# logger.setLevel(logging.INFO)


class WebcamPredictor:
    """The main class in the virtual walking project. When models have been trained, it loads all these models
    and orchestrates everything. It controlls the acquisition of the frames from the webcam, the creation of the
    frame groups, the Google Maps Controller and the output from the webcam. After it's initialised, predictor is its
    main class.
    
    Returns:
        WebcamPredictor: The predictor.
    """
    def __init__(self, config_path=None, coordinates=None):
        """WebcamPredictor class constructor.
        
        Args:
            config_path (str, optional): Path of the config.yaml file. If none, ./config_resnet.yml is used.
            coordinates (tuple, optional): Coordinates for the initialization of the map. If None, the walk starts
            Zaragoza (Spain).
        """
        if config_path == None:
            config_path = Path(__file__).parents[1].joinpath("config_resnet.yml")
        else:
            config_path = Path(config_path)

        with open(str(config_path)) as file:
            config = yaml.full_load(file)
 
        self.classes = config["classes"]
        self.n_frames = 5
        self.threshold_nn = config["threshold_nn"]
        output_video_dim = config["output_video_dim"]
        default_limit = config["default_limit"]
        driver_path = config.get("driver_path", None)
        
        self.model = config["model"]
        self.backbone = config["backbone"]
        # self.backbone = backbone
        self.output_stride = config["posenet_stride"]
        logging.info('Using {} model'.format(self.model))

        if self.model == 'LSTM':
            if config["paths"].get("LSTM", False):
                LSTM_PATH = Path(config["paths"].get("LSTM"))
            else:
                LSTM_PATH = Path(__file__).parents[1].joinpath('models/LSTM.h5')

            self.model_lstm = LSTMModel(
                self.classes,
                input_dim=28,
                load_path_NN=str(LSTM_PATH)
            )
            self.process_frames = self.process_list_lstm
        else:
            if config["paths"].get("PCA", False):
                PCA_PATH = Path(config["paths"].get("PCA"))
            else:
                PCA_PATH = Path(__file__).parents[1].joinpath('models/PCA.pkl')

            if config["paths"].get("NN", False):
                NN_PATH = Path(config["paths"].get("NN"))
            else:
                NN_PATH = Path(__file__).parents[1].joinpath('models/NN.h5')

            if config["paths"].get("SCALER", False):
                SCALER_PATH = Path(config["paths"].get("SCALER"))
            else:
                SCALER_PATH = Path(__file__).parents[1].joinpath('models/SCALER.pkl')

            self.model = FullModel(
                classes=self.classes,
                load_path_PCA=str(PCA_PATH),
                load_path_NN=str(NN_PATH),
                load_path_scaler=str(SCALER_PATH)
            )
            self.process_frames = self.process_list

        if config["paths"].get("posenet", False):
            
            POSE_PATH = config["paths"].get("posenet", False)
            if self.backbone == 'resnet':
                input_dim = (256, 200)
                rescale = output_video_dim[0] / input_dim[0], output_video_dim[1] / input_dim[1]
            else:
                rescale = output_video_dim[0] / 257, output_video_dim[1] / 257
        else:
            if self.backbone == 'resnet':
                POSE_PATH = Path(__file__).parents[1].joinpath('models/resnet_stride{}/model-stride{}.json'.format(self.output_stride, self.output_stride))
                input_dim = (256, 200)
                rescale = output_video_dim[0] / input_dim[0], output_video_dim[1] / input_dim[1]
            else:
                POSE_PATH = Path(__file__).parents[1].joinpath('models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')
                rescale = output_video_dim[0] / 257, output_video_dim[1] / 257

        self.processor = DataProcessor(POSE_PATH, rescale=rescale, backbone=self.backbone, output_stride=self.output_stride)

        if coordinates is not None:
            self.controller = Controller(self.classes, coordinates=coordinates, driver_path=driver_path,
                                         time_rotation=config["time_rotation"])
        else:
            self.controller = Controller(self.classes, time_rotation=config["time_rotation"])

        initial_time = time.time()
        self.last_calls = {element: [initial_time, default_limit] for element in self.classes}

        self.font, self.color = self._prepare_painter()

    def _prepare_painter(self):
        font = cv2.FONT_HERSHEY_PLAIN
        color = (131, 255, 51)
        return font, color

    def predictor(self, output_dim=None, show_skeleton=False, times_v=1):
        """The main method in this class. It processes the input form the webcam,
        creating the groups of frames from which the action will be predicted.
        The selection is based on whether each frame is suitable as a first frame
        in the for the prediction.

        When it finds a proper group of 5 frames, it processes it.
        
        Args:
            output_dim (tuple, optional): The output dim of the webcam output. If None,
            the original size of the video is taken.
            show_skeleton (bool, optional): Whether skeleton should be shown in the image or not.
            Defaults to False.
            times_v (int, optional): Currently not used. Defaults to 1.
        """
        probabilities = None
        capture = cv2.VideoCapture(0)
        if output_dim is None:
            output_dim = (int(capture.get(4)), int(capture.get(3)))

        buffer = []
        buffer_og = []  # For populating future buffers
        valid = 0
        while True:
            # _, frame_orig = capture.read()
            # frame = cv2.resize(frame_orig, network_frame_size, interpolation=cv2.INTER_LINEAR)
            _, frame = capture.read()
            person = self.processor.process_live_frame(frame)

            if valid == 0 and person.is_valid_first():
                # frame = cv2.resize(frame, output_dim[::-1], interpolation=cv2.INTER_LINEAR)
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
                person.draw_points(frame)
                self._write_probabilities(frame, probabilities)
                self._write_distance(frame, self.controller.distance_calculator.distance)
                cv2.imshow('frame', frame)

            # End of while
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def process_list(self, buffer, times_v):
        person_movement = PersonMovement(buffer, times_v, model = self.model)
        logging.info("Shape {}".format(person_movement.coords.shape))

        prediction, probabilities = self.model.predict(person_movement.coords, self.threshold_nn)
        prediction = prediction[0]
        probabilities = probabilities[0]
        if time.time() - self.last_calls[prediction][0] > self.last_calls[prediction][1]:
            self.last_calls[prediction][0] = time.time()
            self.controller.perform_action_name(prediction)

        return probabilities

    def process_list_lstm(self, buffer, *args):
        """Processs a list of frames with LSTM.
        
        Args:
            buffer (list): List of persons extracted from frames.
        
        Returns:
            list: Probabilities for each action
        """
        person_movement = PersonMovement(buffer, model='LSTM').coords
        prediction, probabilities = self.model_lstm.predict_NN(person_movement, self.threshold_nn)
        prediction = prediction[0]
        probabilities = probabilities[0]
        if time.time() - self.last_calls[prediction][0] > self.last_calls[prediction][1]:
            self.last_calls[prediction][0] = time.time()
            self.controller.perform_action_name(prediction)

        return probabilities

    def _write_probabilities(self, frame, probabilities):
        """Write probabilities for each class in the output frame
        
        Args:
            frame (ndarray): Array containing the image
            probabilities (list): List containing the probability for each action.
        
        Returns:
            ndarray: Array containing the image with the new text.
        """
        font = cv2.FONT_HERSHEY_PLAIN
        color = (131, 255, 51)
        for i, (p, c) in enumerate(zip(probabilities, self.classes)):
            pos = (10, 20 * (i + 1) + 50)
            cv2.putText(frame, '{}: {:.3f}'.format(c, p), pos, font, 0.8, color, 1)
        return frame

    def _write_distance(self, frame, distance):
        """Writes and formats the distance walked in the output frame.
        
        Args:
            frame ([ndarray): Array containing the image
            distance (float): Distance made during the walk
        
        Returns:
            ndarray: Array containing the image with the new text.
        """
        font = cv2.FONT_HERSHEY_PLAIN
        color = (131, 255, 51)
        # pos = (10, 20 * (4 + 1) + 50)
        pos = (10, 400)
        if distance < 500:
            cv2.putText(frame, 'Distance: {} m'.format(int(distance)), pos, font, 1, color, 1)
        else:
            cv2.putText(frame, 'Distance: {:.2f} km'.format(distance/1000), pos, font, 1, color, 1)
        return frame
