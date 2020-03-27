import tensorflow as tf
import numpy as np
from utils.person import Person

def load_model(path):
    model = tf.lite.Interpreter(path)
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    return model, input_details, output_details

def prepare_frame(frame, input_dim):
    frame = tf.reshape(tf.image.resize(frame, input_dim), [1, input_dim[0], input_dim[1], 3])
    frame = (np.float32(frame) - 127.5) / 127.5
    return frame

def prepare_list_frames(frames, input_dim):
    return [prepare_frame(frame, input_dim) for frame in frames]

def get_model_output(model, frame, input_details, output_details):
    model.set_tensor(input_details[0]['index'], frame)
    model.invoke()

    output_data = np.squeeze(model.get_tensor(output_details[0]['index']))
    offset_data = np.squeeze(model.get_tensor(output_details[1]['index']))

    return output_data, offset_data

def read_labels_txt(path, actions):
    dict_frames = {}
    curr_label = ''
    with open(path, 'r') as F:
        for line in F.readlines():
            if line.replace(' ', '') == '\n':
                continue
            elif line.split('_')[0] in actions:
                curr_label = line.split('_')[0]
                if curr_label not in dict_frames.keys():
                    dict_frames[curr_label] = []
            else:
                frames = [int(x) for x in line.replace('\n', '').split(' ')]
                frames.sort()
                dict_frames[curr_label].append(frames)

    return dict_frames