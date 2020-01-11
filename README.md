# Object detection: Tracking Rats in Open Field Videos

This project tracks the movement of rats in open field videos using the [Mask R-CNN](https://github.com/matterport/Mask_RCNN) repository and a Mask R-CNN model *mask_rcnn_rat_cfg_0005.h5*



<p align="center">
  <img width="500" height="300" src="https://github.com/nhalle/rat-openfield-maskrcnn/blob/master/example.gif">
</p>

<p align="center">
  <img src="https://github.com/nhalle/rat-openfield-maskrcnn/blob/master/example_path_result.png">
</p>

## Description

The Open Field Maze(OFM) is a common method to study loco-motor ability and anxiety-like behavior in rats, and thus being used for various psychology research. Automating behavioral observations is necessary to enable researchers to study behavior in more reliable and consistent ways and allow experiments to be conducted in longer periods of time. Therefore, this project used the Mask-Region-based Convolutional Neural Network model (Mask-RCNN) to develop a reliable program that tracks rat movement in the OFM.


## Installation

### Requirements

This project requires python 3.4, tensorflow version 1.14 (tensorflow 2.0 is not compatible with the Mask_RCNN library unfortunately), and Keras 2.3.1. We recommend using a virtual environment such as pip's [venv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

### Instructions

1. Clone this repository

2. Setup pip virtual environment

           python3 -m venv venv
            
           source venv/bin/activate

2. Install dependencies

           pip install -r requirements.txt

3. Download Model 
      
           (in the process of having a download link here)

4. Download the dataset
      
           (in the process of having a download link here)


## Usage

To run the open field video analyzer:
python analyze_video.py <path of video> <video name> <number of frames>

ex.
python analyze_video.py ./video/Video1.mp4 test1 20


## Credits

First we'd like to thank the psychology department of Franklin and Marshall college for providing the video of OFM. We would also like to thank the authors who providing tutorials introducing how to use the mask R-CNN model in keras with tensorflow object detection platform

### Authors: 
Kitty Chen & Noah Halle


## License:
