import argparse
import json
import logging
import os
import posixpath
import re
import shutil
import urllib.request
import zlib

import cv2
import numpy as np
import requests
import tensorflow as tf
import tfjs_graph_converter.api as tfjs
import tfjs_graph_converter.util as tfjsutil

logger = logging.getLogger(__name__)

def load_model_mobilenet(path):
    model = tf.lite.Interpreter(path)
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    return model, input_details, output_details

def prepare_frame_mobilenet(frame, *args):
    input_dim = (257, 257)
    frame = tf.reshape(tf.image.resize(frame, input_dim), [1, input_dim[0], input_dim[1], 3])
    frame = (np.float32(frame) - 127.5) / 127.5
    return frame

def prepare_frame_resnet(frame, input_dim):
    frame = tf.reshape(tf.image.resize(frame, input_dim), [1, input_dim[0], input_dim[1], 3])
    return frame.numpy()

def prepare_list_frames(frames, input_dim):
    # I think this function is not used, so it takes resnet by default.
    return [prepare_frame_resnet(frame, input_dim) for frame in frames]

def get_model_output_resnet(sess, frame, input_details, output_details):
    results = infer_model(frame, sess, input_details, output_details)
    offset_data = np.squeeze(results[2], 0)
    output_data = np.squeeze(results[3], 0)
    return output_data, offset_data

def get_model_output_mobilenet(model, frame, input_details, output_details):
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

def infer_model(img, sess, input_tensor, output_tensor_names):
    results = sess.run(output_tensor_names, feed_dict={input_tensor: img})
    return results

def get_tensors_graph(graph):
    input_tensor_names = tfjsutil.get_input_tensors(graph)
    output_tensor_names = tfjsutil.get_output_tensors(graph)
    input_tensor = graph.get_tensor_by_name(input_tensor_names[0])

    return input_tensor, output_tensor_names


def load_model_resnet(model_path):
    graph = tfjs.load_graph_model(model_path)
    sess = tf.compat.v1.Session(graph=graph)
    return sess, graph

def fix_model_file(model_cfg):
    '''
    Function from https://github.com/atomicbits/posenet-python/blob/eef813ad16488812817b344afa1f390f0c22623d/posenet/converter/tfjsdownload.py
    '''
    model_file_path = os.path.join(model_cfg['tfjs_dir'], model_cfg['filename'])

    if not model_cfg['filename'] == 'model.json':
        # The expected filename for the model json file is 'model.json'.
        # See tfjs_common.ARTIFACT_MODEL_JSON_FILE_NAME in the tensorflowjs codebase.
        normalized_model_json_file = os.path.join(model_cfg['tfjs_dir'], 'model.json')
        shutil.copyfile(model_file_path, normalized_model_json_file)

    with open(model_file_path, 'r') as f:
        json_model_def = json.load(f)

    return json_model_def


def download_single_file(base_url, filename, save_dir):
    '''
    Function from https://github.com/atomicbits/posenet-python/blob/eef813ad16488812817b344afa1f390f0c22623d/posenet/converter/tfjsdownload.py
    '''
    output_path = os.path.join(save_dir, filename)
    url = posixpath.join(base_url, filename)
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    if response.info().get('Content-Encoding') == 'gzip':
        data = zlib.decompress(response.read(), zlib.MAX_WBITS | 32)
    else:
        # this path not tested since gzip encoding default on google server
        # may need additional encoding/text handling if hit in the future
        data = response.read()
    with open(output_path, 'wb') as f:
        f.write(data)


def download_tfjs_model(model_cfg):
    """
    Function from https://github.com/atomicbits/posenet-python/blob/eef813ad16488812817b344afa1f390f0c22623d/posenet/converter/tfjsdownload.py
    Download a tfjs model with saved weights.
    :param model_cfg: The model configuration
    """
    model_file_path = os.path.join(model_cfg['tfjs_dir'], model_cfg['filename'])
    if os.path.exists(model_file_path):
        print('Model file already exists: %s...' % model_file_path)
        return
    if not os.path.exists(model_cfg['tfjs_dir']):
        os.makedirs(model_cfg['tfjs_dir'])

    download_single_file(model_cfg['base_url'], model_cfg['filename'], model_cfg['tfjs_dir'])

    json_model_def = fix_model_file(model_cfg)

    shard_paths = json_model_def['weightsManifest'][0]['paths']
    for shard in shard_paths:
        download_single_file(model_cfg['base_url'], shard, model_cfg['tfjs_dir'])

def download_file_from_google_drive(id, destination):
    '''
    Taken from the following StackOverflow answer: https://stackoverflow.com/a/39225039
    '''
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    '''
    Taken from the following StackOverflow answer: https://stackoverflow.com/a/39225039
    '''
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    '''
    Taken from the following StackOverflow answer: https://stackoverflow.com/a/39225039
    '''
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
