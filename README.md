# Virtual walks in Google Street View

_Para la versión en español, [haz click aquí](README_ES.md)._

During the quarantine, we're currently experiencing due to the COVID-19 pandemic our rights to move freely on the street are trimmed in favour of the common wellbeing. People can only go out in certain situations like doing the grocery. Many borders are closed and travelling is almosy totally banned in most countries.

_Virtual Walks_ is a project that uses Pose Estimation models along with LSTM neural networks in order to simulate walks in Google Street View. For pose estimation, [PoseNet](https://www.tensorflow.org/lite/models/pose_estimation/overview) model has been adapted, while for the action detection part, an LSTM model has been developed using [TensorFlow 2.0](https://www.tensorflow.org/).

This project is capable of simulating walking around the street all over the world with the help of [Google Street View](https://www.google.com/intl/es_ES/streetview/).

Tensorflow 2.0, Selenium and Python 3.7 are the main technologies used in this project.

## How does it work

PoseNet has been combined with an LSTM model to infer the action that the person is performing. Once the action is detected it is pased to the controller; the part that interacts with Google Street View.

1. A Selenium Firefox window is opened.
1. Using the webcam, the system takes photos of the person who will be making one of the four main actions used for walking:
    
    * Stand
    * Walk
    * Turn right
    * Turn left

1. For each photo taken, PoseNet is used to infer the position of the joints in the image.
1. Groups of 5 frames are made, starting from a frame that has to meet certain considerations of confidence in the joints detected. Missing joint inference is made in frames behind 1st one.
1. Each group of frames is passed to a LSTM model with a FF Neural Network attached after it and an action is predicted.
1. The predicted action is passed to the selenium controller and brings the action to reality in the opened Firefox Window

Currently, there is another model that can be used to run this program. Instead of a LSTM, joint velocities are calculated across the frames in the 5-frame groups and passed along with the joint positions to a PCA and FF Neural Network to predict the action. The default model is the LSTM, as we consider it the methodologically correct one and is the model with the highest precission.

As the action prediction could be (depending on the host computer's specifications) much faster than the average walking speed, an action can be only executed once every 0.5 seconds. This parameter is customizable.

## Use case example

As it can be seen in the image, the skeleton is inferred form the image and an action is predicted and executed.

![Example walk in Paris](./readme_resources/Paris.gif)

## Installation and use
Remember that a Webcam is needed to use this program, as actions are predicted from the frames taken with it.

It is recommended to install it in a new Python 3.7 environment to avoid issues and version conflicts.

Install tensorflowjs, required to run ResNet:
```
pip install tensorflowjs
```

Clone and install tensorflowjs graph model converter, following the steps in [tfjs-to-tf](https://github.com/patlevin/tfjs-to-tf)

Clone the git repository

```
git clone https://github.com/Moving-AI/virtual-walk.git
```

Install dependencies by running

```
pip install -r requirements.txt
```

Install Firefox and download [Geckodriver](https://github.com/mozilla/geckodriver/releases). Then specify the path in 
[config_resnet.yml](./config_resnet.yml) under the "driver_path" option.

Download the used models by running the [download_models](./download_models.py) file. This script will download PoseNet
models (MobileNet and ResNet with both output strides, 16 and 32), LSTM, PCA, scaler and neural network. The link to
download the models separately can be found below. 

```
cd virtual-walk
python3 download_models.py
```

Finally, you can run [execute.py](./execute.py) to try it.

```
python3 execute.py
```

Considerations during usage: 

- Our experience using the model tells us that a slightly bright enviroment is preferred rather than a very bright one.

- The system is sensitive to the position of the webcam.

To sum up, a position close to the one shown in the GIF should be used.

#### Links to our models
- [LSTM](https://drive.google.com/uc?export=download&id=1JydPMY58DVZr3qcZ3d7EPZWfq__yJH2Z)
- [Scaler](https://drive.google.com/uc?export=download&id=1eQUYZB1ZTWRjXH4Y-gxs2wsgAK30iwgC)
- [PCA](https://drive.google.com/uc?export=download&id=1cYMuGlfBdkbH6wd9x__1D07I64VA94wE)
- [Feed-forward neural network](https://drive.google.com/uc?export=download&id=1dn51tNt96cWesufjCRtuQJQd2S3Ro6fu)

### Training

Probably the training part is the weakest in this project, due to our lack of training data and computing power. Our training data generation process consisted on 40 minutes of recordings. In each video, one person appeared making one specific action for a certain period of time. As it will be discussed in the next steps section, our models tend to overfit in spite of having a working system. An example of the training data can be seen below.

<img src="/readme_resources/Walking.gif" height="150"> 

The models we have trained and the ones from which the examples have been generated can be downloaded running the [download_models](./download_models.py) file. In the images below the training performance is shown:



<img src="./readme_resources/epoch_categorical_accuracy.svg" height="200" hspace="20" />  |  <img src="./readme_resources/epoch_loss.svg" height="150" hspace="50" />
-------------------------|------------------------



If someone wants to train another LSTM model, the [DataProcessor](./source/dataprocessing/__init__.py) class is provided. It can process the videos located in a folder, reading the valid frame numbers from a labels.txt file and generating a CSV file with the training examples. This file can be used in [train.py](./train.py) to generate a new LSTM model. The path for this model would be passed to the [WebcamPredictor](./source/webcam_predictor.py) class and the system would use this new model.

## Next steps

- Generating more training data. In this project we have tried to get what could be considered a MVP, robustness has never been a main goal. As it can be seen in the Training section, the model does not appear to overfit, even knowing that LSTM tend very much to overfit. However, the training and testing data are very similar, as the videos are people making "loop" actions. So we expect the model to have underlying overfitting that cannot be detected witout more videos. Probably, recording more videos in different light conditions would make the model more robust and consistent.

- Turning to the right and to the left are not predicted with the same accuracy in spite of being symmetric actions. A specular reflection of the coordinates could be used to be more consistent in the turn predictions.


## Authors

* **Javier Gamazo** - [Github](https://github.com/javirk). [LinkedIn](https://www.linkedin.com/in/javier-gamazo-tejero/)

* **Gonzalo Izaguirre** - [Github](https://github.com/gontxomde). [LinkedIn](https://www.linkedin.com/in/gizaguirre/)

## License

This project is under MIT license. See [LICENSE](LICENSE) for more details.

## Acknowledgments

- [@atomicbits](https://github.com/atomicbits) for the [repo](https://github.com/atomicbits/posenet-python/) with the tools used in order to download the TFJS models of PoseNet to be used in TensorFlow.
- [@tensorflow](https://github.com/tensorflow/) for [Posenet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) models.
- [@patlevin](https://github.com/patlevin/tfjs-to-tf) for the tools used for creating a session in Python from the graph files.
- [@ajaichemmanam](https://github.com/ajaichemmanam/simple_posenet_python) for the help provided when struggling with opening graph files with Python.
- [@felixchenfy](https://github.com/felixchenfy) and his repo [Realtime-Action-Recognition](https://github.com/felixchenfy/Realtime-Action-Recognition)
for the inspiration.
