from utils.model import FullModel
from utils.dataprocessor import DataProcessor 
from utils.person_frames import PersonMovement

from pathlib import Path
import cv2
from copy import deepcopy

class WebcamPredictor():

    def __init__(self, pca_model_path = None, nn_model_path = None, pose_model_path = None):

        self.n_frames = 5

        if pca_model_path is not None:
            PCA_PATH = Path(pca_model_path)
        else:
            PCA_PATH = Path(__file__).parents[1].joinpath("models/PCA.pkl")
        if nn_model_path is not None:
            NN_PATH = Path(nn_model_path)
        else:
            NN_PATH = Path(__file__).parents[1].joinpath("models/NN.h5")

        if pose_model_path is None:
            POSE_PATH = Path(__file__).parents[1].joinpath("models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")
        else:
            POSE_PATH = pose_model_path

        self.model = FullModel(
            classes = ["walk", "stand", "left", "right"],
            load_path_PCA = str(PCA_PATH),
            load_path_NN = str(NN_PATH)
        )
        #print("creo processor")
        self.processor = DataProcessor(POSE_PATH)


    def predictor(self, output_dim  =None):
        
        network_frame_size = (257, 257)
        capture = cv2.VideoCapture(0)
        if output_dim is None:
            output_dim = (int(capture.get(4)), int(capture.get(3)))

        buffer = []
        buffer_og = [] # For populating future buffers
        valid = 0
        while True:
            _, frame = capture.read()
            frame = cv2.resize(frame, network_frame_size,  interpolation=cv2.INTER_LINEAR)
            
            person = self.processor.process_live_frame(frame)

            if valid == 0 and person.is_valid_first():
                frame = cv2.resize(frame, output_dim[::-1],  interpolation=cv2.INTER_LINEAR)
                cv2.imshow("WebCam", frame)
                buffer.append(person)
                buffer_og.append(person)
                valid += 1

            elif 0 < valid < self.n_frames -1 and person.is_valid_other():
                
                # If valid as first, take into account for future frames
                if person.is_valid_first():
                    buffer_og.append(deepcopy(person))
                else:
                    buffer_og.append(False)

                person.infer_lc_keypoints(buffer[valid - 1])

                buffer.append(person)
                valid += 1
            elif valid == self.n_frames -1 and person.is_valid_other():
                # Here is the ONLY case in which we process a group of frames

                # If frame was valid for first initially, take into account for future frames
                if person.is_valid_first():
                    buffer_og.append(deepcopy(person))
                else:
                    buffer_og.append(False)
                person.infer_lc_keypoints(buffer[valid -1])

                buffer.append(person)
                
                self.process_list(buffer)

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
                valid += 1
            
            else:
                buffer = []
                valid = 0
            

            #End of while
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



    def process_list(self, buffer):
        person_movement = PersonMovement(buffer)

        print(type(person_movement.coords))
            
            
            





