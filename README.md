# Drone_Badminton_Track

# The code for player tracking with yolo v5 and deepsort
It worked well in badminton doubles, but may not work well for more complex MOT. 

## Requirements
- python 3.8
- When creating a new virtual environment, run `pip install -r requirements.txt`

## Usage

- input video source: `--source badminton_example`
- object detection model: `--yolo_model ./model/yolov5x.pt`
- tracking model: `--deep_sort_model ./model/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth` 
- image size: `--img 1280`, `--img 6230 1072`, etc. 
- save video tracking results: `--save-vid`
- save MOT compliant results to *.txt: `--save-txt`
- output folder: `--project ../yolov5_deepsort_results/badminton_doubles`



## Acknowledgements:
For this project, we relied on research codes from:https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet
