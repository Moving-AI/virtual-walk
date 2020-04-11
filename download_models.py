import json
import os
import posixpath
import shutil
import urllib.request
import zlib

'''
EVERYTHING FROM https://github.com/atomicbits/posenet-python/blob/eef813ad16488812817b344afa1f390f0c22623d/posenet/converter/tfjsdownload.py
'''

def fix_model_file(model_cfg):
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


model_cfg = {}
model_cfg['tfjs_dir'] = 'model/stride_32/'
model_cfg['filename'] = 'model-stride32.json'
model_cfg['base_url'] = 'https://storage.googleapis.com/tfjs-models/savedmodel/posenet/resnet50/float/'

download_tfjs_model(model_cfg)
