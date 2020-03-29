# Virtual walks in Google Street View

## Introduction

In the current situation of a COVID-19 worldwide pandemia, walking in the streets is banned if there is not a major reason (at least in Spain). In this project we try to solve this problem by simulating a walk in Google Street View using the webcam.

We use [TFLite Pose Estimation model] (https://www.tensorflow.org/lite/models/pose_estimation/overview) to get the coordinates of the body joints. With this coordinates we use a classifier to distinguish between the different possible actions (currently stand, walk, turn right, turn left) and a Python interface to interact with Google Street View.

With this tools, it is possible to walk around the street while pretending to walk, at home.

## Examples
TODO

## Installation
TODO

