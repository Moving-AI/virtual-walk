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

TODO

## Next steps

TODO

## Authors

* **Javier Gamazo** - [Github](https://github.com/javirk). [LinkedIn](https://www.linkedin.com/in/javier-gamazo-tejero/)

* **Gonzalo Izaguirre** - [Github](https://github.com/gontxomde). [LinkedIn](https://www.linkedin.com/in/gizaguirre/)

## License

This project is under MIT license. See [LICENSE](LICENSE) for more details.

## Acknowledgments

* [zzh8829](https://github.com/zzh8829/yolov3-tf2) for YOLO's code
* [Tensorflow](https://www.tensorflow.org/) for Pix2Pix' code

