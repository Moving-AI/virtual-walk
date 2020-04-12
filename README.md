# Virtual walks in Google Street View

During the quarantine we're currently in due to the COVID-19 pandemia our rights to move freely across the streets are trimmed in favour of the common wellbeing. People can only move in certain situations like doing the grocery. Most borders are closed and travelling is almosy totally banned in most countries.

_Virtual Walks_ is a project that uses Pose Estimation models along with LSTM neural networks in order to simulate walks in Google Street View. For pose estimation, [PoseNet](https://www.tensorflow.org/lite/models/pose_estimation/overview) model has been adapted, while for the action detection part, a LSTM model has been created using (TensorFlow 2.0)[https://www.tensorflow.org/].

This project is capable of simulating walking around the street all over the world with the help of (Google Street View)[https://www.google.com/intl/es_ES/streetview/].

Tensorflow 2.0, Selenium and Python 3.7 are the main technologies used in this project.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/1200px-TensorFlowLogo.svg.png" data-canonical-src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/1200px-TensorFlowLogo.svg.png" height="200" hspace="20" />  |  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Google_Street_View_icon.svg/1200px-Google_Street_View_icon.svg.png" data-canonical-src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Google_Street_View_icon.svg/1200px-Google_Street_View_icon.svg.png" height="150" hspace="50" />
-------------------------|-------------------------


## How does it work

PoseNet has been combined with a LSTM model to infer the action that is being made by the person. Once the action is detected it is pased to the controller; the part that interacts with Google Street View.

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

As the action prediction could be (depending on the host computer's specifications) quite faster than the average walking speed, an action can be only executed once every 0.5 seconds.

## Use case example

As it can be seen in the image, the skeleton is inferred form the image and an action is predicted and executed.

![Example walk in Paris](./readme_resources/Paris.gif)

## Installation

TODO

### Training

Probably the training part is the weakest part in this project, due to our lack of training data and computing power. Our training data generation process consisted on 40 minutes of recordings. In each video, one person appeared making one specific action for a certain period of time. As it will be discussed in the next steps section, our models tend to overfit in spite of having a working system.

The models we have trained and the ones from which the examples have been generated can be downloaded running the [download_models](./download_models.py) file.

If someone wants to train another LSTM model, the (DataProcessor)[./source/dataprocessing/__init__.py] class is provided. It can process the videos located in a folder, reading the valid frame numbers from a labels.txt file and generating a CSV file with the training examples. This file can be used in (train.py)[./train.py] to generate a new LSTM model. The path for this model would be passed to the (WebcamPredictor)[./source/webcam_predictor.py] class and the system would use this new model.

## Next steps

TODO

## Authors

* **Javier Gamazo** - [Github](https://github.com/javirk). [LinkedIn](https://www.linkedin.com/in/javier-gamazo-tejero/)

* **Gonzalo Izaguirre** - [Github](https://github.com/gontxomde). [LinkedIn](https://www.linkedin.com/in/gizaguirre/)

## License

This project is under MIT license. See [LICENSE](LICENSE) for more details.

## Acknowledgments

- @(atomicbits)[https://github.com/atomicbits] for the (repo)[https://github.com/atomicbits/posenet-python/] with the tools used in order to download the TFJS models of PoseNet to be used in TensorFlow.
- @(tensorflow)[https://github.com/tensorflow/] for (Posenet)[https://github.com/tensorflow/tfjs-models/tree/master/posenet] models.
- @(patlevin)[https://github.com/patlevin/tfjs-to-tf] for the tools used for creating a session in Python from the graph files.
- @(ajaichemmanam)[https://github.com/ajaichemmanam/simple_posenet_python] for the help provided when struggling with opening graph files with Python.
