import cv2
import numpy as np
import tensorflow as tf

import utils.funciones as f
from utils.person import Person

path = r'./models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
model, input_details, output_details = f.load_model(path)
INPUT_DIM = (257, 257)
original = (640, 480)

def video(output_dim=INPUT_DIM):
    cap = cv2.VideoCapture(0)
    rescale = output_dim[0] / INPUT_DIM[0], output_dim[1] / INPUT_DIM[1]
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame2 = cv2.resize(frame, output_dim, interpolation=cv2.INTER_LINEAR)
        frame = f.prepare_frame(frame, INPUT_DIM)
        output_data, offset_data = f.get_model_output(model, frame, input_details, output_details)

        p = Person(output_data, offset_data, rescale, threshold=0.5)

        img = p.draw_points(frame2)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', frame2)
        # cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def imagen():
    rescale=(1,1)
    img = cv2.imread('Screenshot_1.png')
    img2 = cv2.resize(img, INPUT_DIM, interpolation=cv2.INTER_LINEAR)
    img = tf.reshape(tf.image.resize(img, INPUT_DIM), [1, INPUT_DIM[0], INPUT_DIM[1], 3])
    img = (np.float32(img) - 127.5) / 127.5
    model.set_tensor(input_details[0]['index'], img)
    model.invoke()
    output_data = np.squeeze(model.get_tensor(output_details[0]['index']))
    offset_data = np.squeeze(model.get_tensor(output_details[1]['index']))
    p = Person(output_data, offset_data, rescale)
    p.draw_points(img2)
    cv2.imshow("{}".format(p.confidence()), img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    p.skeleton_to_txt('prueba.txt')
    return p

def from_txt(path):
    img = cv2.imread('messi.jpg')
    img2 = cv2.resize(img, INPUT_DIM, interpolation=cv2.INTER_LINEAR)
    p = Person(path_txt=path)
    p.draw_points(img2)
    return p


if __name__ == '__main__':
    # video(original)
    p = imagen()
    # path = 'prueba.txt'
    # p = from_txt(path)
