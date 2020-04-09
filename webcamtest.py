import cv2
import numpy as np
import tensorflow as tf
import tfjs_graph_converter as tfjs

import source.funciones as f
from source.entities.person import Person

path = r'./models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
# model, input_details, output_details = f.load_model(path)
INPUT_DIM = (257, 257)
original = (640, 480)


def infer_model(img, sess, input_tensor, output_tensor_names):
    results = sess.run(output_tensor_names, feed_dict={input_tensor: img})
    return results

def get_tensors_graph(graph):
    input_tensor_names = tfjs.util.get_input_tensors(graph)
    output_tensor_names = tfjs.util.get_output_tensors(graph)
    input_tensor = graph.get_tensor_by_name(input_tensor_names[0])

    return input_tensor, output_tensor_names


def load_model(model_path):
    return tfjs.api.load_graph_model(model_path)


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

def video_resnet(output_dim=INPUT_DIM):
    output_stride = 32
    cap = cv2.VideoCapture(0)
    graph = load_model('models/resnet_stride32/model-stride32.json')
    input_tensor, output_tensor_names = get_tensors_graph(graph)
    sess = tf.compat.v1.Session(graph=graph)
    img_width, img_height = (480, 640)
    target_width = (int(img_width) // output_stride) * output_stride + 1
    target_height = (int(img_height) // output_stride) * output_stride + 1
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()
        # frame2 = cv2.resize(frame, output_dim, interpolation=cv2.INTER_LINEAR)
        frame2 = tf.reshape(tf.image.resize(frame, (target_width, target_height)), [1, target_width, target_height, 3])
        res = infer_model(frame2.numpy(), sess, input_tensor, output_tensor_names)
        offsets = np.squeeze(res[2], 0)
        heatmaps = np.squeeze(res[3], 0)

        p = Person(heatmaps, offsets, output_stride=output_stride, threshold=0.5)
        frame = p.draw_points(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)
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

def imagen_resnet():
    output_stride = 32
    img = cv2.imread('messi.png')
    ## PREPARE IMAGE
    img_width, img_height = img.shape[:2]
    target_width = (int(img_width) // output_stride) * output_stride + 1
    target_height = (int(img_height) // output_stride) * output_stride + 1
    img2 = tf.reshape(tf.image.resize(img, (target_width, target_height)), [1, target_width, target_height, 3])

    ## FEED TO NETWORK
    graph = load_model(f'models/resnet_stride{output_stride}_q2/model-stride{output_stride}.json')
    input_tensor, output_tensor_names = get_tensors_graph(graph)
    sess = tf.compat.v1.Session(graph=graph)
    res = infer_model(img2.numpy(), sess, input_tensor, output_tensor_names)

    offsets = np.squeeze(res[2], 0)
    heatmaps = np.squeeze(res[3], 0)

    p = Person(heatmaps, offsets, output_stride=output_stride, threshold=0.5)
    p.draw_points(img)
    cv2.imshow("{}".format(p.confidence()), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return p

def from_txt(path):
    img = cv2.imread('messi.jpg')
    img2 = cv2.resize(img, INPUT_DIM, interpolation=cv2.INTER_LINEAR)
    p = Person(path_txt=path)
    p.draw_points(img2)
    return p


if __name__ == '__main__':
    # video(original)
    #p = imagen()
    # path = 'prueba.txt'
    # p = from_txt(path)
    p=video_resnet()
