import cv2
import utils.funciones as f
import os
from utils.person import Person


INPUT_DIM = (257, 257)
output_dim = (480, 640)
N_FRAMES = 5
THRESHOLD = 0.5
images_data = './data/images'
labels_data = 'data/labels/'
cap = cv2.VideoCapture(0)
rescale = output_dim[0] / INPUT_DIM[0], output_dim[1] / INPUT_DIM[1]
path_model = r'./models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'

model, input_details, output_details = f.load_model(path_model)

folders_data = [o for o in os.listdir(images_data) if os.path.isdir(os.path.join(images_data, o))]

for folder in folders_data:
    label = folder.split('_')[0]  # Folders must have ${label}_${num} format. Where num is just an identifier
    images_names = [i for i in os.listdir(os.path.join(images_data, folder))]
    images = [i[:-4] for i in images_names]
