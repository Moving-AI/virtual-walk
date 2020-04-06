import argparse
import logging
import os
import re

import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

def load_model(path):
    model = tf.lite.Interpreter(path)
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    return model, input_details, output_details

def prepare_frame(frame, input_dim):
    frame = tf.reshape(tf.image.resize(frame, input_dim), [1, input_dim[0], input_dim[1], 3])
    
    #print(frame.shape)
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
    regex = r"[a-z]+_[0-9]+"
    with open(path, 'r') as F:
        for line in F.readlines():
            if line.replace(' ', '') == '\n':
                continue
            elif line.split('_')[0] in actions:
                curr_label = re.search(regex, line).group(0)
                if curr_label not in dict_frames:
                    dict_frames[curr_label] = []
            else:
                frames = [int(x) for x in line.replace('\n', '').split(' ')]
                frames.sort()
                dict_frames[curr_label].append(frames)

    return dict_frames

def process_video(filename, output_shape = (256,256), fps_reduce = 2):
    """Process a video from the resources folder and saves all the frames
    inside a folder with the name of the video
    FILENAME_frame_X.jpg
    
    Args:
        filename (str): Name of the video inside resources
        output_shape (tuple, optional): Size of the output images. Defaults to (256,256).
        fps_reduce (int, optional): Take one image out of  #fps_reduce. 
        Defaults to 2.
    """
    PATH = './resources/{}/'.format(filename.split(".")[0])
    print(PATH)
    try:
        os.mkdir(PATH)
    except:
        os.system("rm -r ./resources/{}".format(filename.split(".")[0]))
        os.mkdir(PATH)

    #Read video
    video = cv2.VideoCapture("./resources/{}".format(filename))
    count=0
    logging.debug("Started reading frames.")
    while video.isOpened():
        logging.debug("Reading frame {}/{} from file {}".format(count+1, video.get(cv2.CAP_PROP_FRAME_COUNT),filename))
        
        #Frame reading, reshaping and saving
        ret,frame = video.read()
        frame = cv2.resize(frame, (256,256))
        if count % fps_reduce == 0:
            cv2.imwrite(PATH+"{}_frame_{}.jpg".format(filename.split(".")[0],count), frame)
        count = count + 1
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            break
    logging.debug("Stop reading files.")
    video.release()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')