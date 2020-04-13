import os
from pathlib import Path

import requests

import source.funciones as f

if __name__ == '__main__':
    output_folder = str(Path(__file__).parents[0].joinpath('models/'))
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        print('The directory is already created.')

    models = {'LSTM.h5': '1JydPMY58DVZr3qcZ3d7EPZWfq__yJH2Z',
              'PCA.pkl': '1cYMuGlfBdkbH6wd9x__1D07I64VA94wE',
              'SCALER.pkl': '1eQUYZB1ZTWRjXH4Y-gxs2wsgAK30iwgC',
              'NN.h5': '1dn51tNt96cWesufjCRtuQJQd2S3Ro6fu'}
    mobilenet_model_name = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'

    print('Downloading MobileNet...', end='\n')
    try:
        mobilenet_destination = str(Path(__file__).parents[0].joinpath(output_folder + mobilenet_model_name))
        session = requests.Session()
        response = session.get('https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite', stream=True)
        f.save_response_content(response, mobilenet_destination)
        print('MobileNet downloaded', end='\n')
    except Exception as e:
        print(f'Error ocurred while downloading MobileNet: {e}', end='\n')


    print('Downloading ResNet...', end='\n')
    for stride in [16, 32]:
        print('Stride ' + str(stride) + '...', end='\n')
        model_cfg = {}
        model_cfg['tfjs_dir'] = str(Path(__file__).parents[0].joinpath(output_folder + 'stride_' + str(stride)))
        model_cfg['filename'] = 'model-stride'+ str(stride)+'.json'
        model_cfg['base_url'] = 'https://storage.googleapis.com/tfjs-models/savedmodel/posenet/resnet50/float/'
        try:
            f.download_tfjs_model(model_cfg)
            print('Stride ' + str(stride) + ' downloaded succesfully', end='\n')
        except Exception as e:
            print(f'Error ocurred while downloading ResNet: {e}', end='\n')

    print('Downloading internal models')
    for filename, file_id in models.items():
        destination = str(Path(__file__).parents[0].joinpath(output_folder + filename))
        f.download_file_from_google_drive(file_id, destination)
