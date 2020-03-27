import cv2
import utils.funciones as f
from utils.person import Person

INPUT_DIM = (257, 257)
output_dim = (480, 640)
N_FRAMES = 5
THRESHOLD = 0.5
cap = cv2.VideoCapture(0)
rescale = output_dim[0] / INPUT_DIM[0], output_dim[1] / INPUT_DIM[1]
path = r'models\posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'

model, input_details, output_details = f.load_model(path)

while True:
    i_frame = 0
    persons = []
    p_previous = None
    while i_frame < 5:
        ret, frame = cap.read()
        frame = f.prepare_frame(frame, INPUT_DIM)
        output_data, offset_data = f.get_model_output(model, frame, input_details, output_details)
        p = Person(output_data, offset_data, threshold=THRESHOLD)
        if p.H == 0:
            # If it is not possible to find the neck or feet. Restart count
            i_frame = 0
            persons = []
            p_previous = p
        else:
            if p_previous is not None:
                low_confidence = p.low_confidence_keypoints()
                for point in low_confidence:
                    p.infer_point(point, p_previous.keypoints[17], p_previous.keypoints[point])
            p_previous = p
            persons.append(p)
            i_frame += 1
