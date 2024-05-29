# Webcam Eye Blink Detection Demo Using Dual Embedding Video Vision Transformer

This repository contains a demo code for running eye blink detection using the model propposed in the WACV 2024 paper "Robust Eye Blink Detection Using Dual Embedding Video Vision Transformer". For detailed information on the model architecture and training process, please visit the paper link or the official implementation repository, or check out the presentation video, links provided below.

### [Official repo📂](https://github.com/hongtuna/Eyeblink-detection-using-ViViT/tree/main) • [Paper📝](https://openaccess.thecvf.com/content/WACV2024/html/Hong_Robust_Eye_Blink_Detection_Using_Dual_Embedding_Video_Vision_Transformer_WACV_2024_paper.html) • [Video 🎥](https://youtu.be/i2CWdyRcgWQ?feature=shared)
## About

Eye blink detection, the task of accurately detecting the timepoint in which a person blinks their eye(s), is generally a simple and easy task for a stationary subject when you have access to equipment such as an IR camera or an eye tracker. However, these equipment tend to show limited performance with the presence of factors such as strong sunlight reflections and non-centered camera angles. Also, for applications such as monitoring eye blinks through webcams (maybe for online classes or meetings), using these kinds of equipment is not an option.

In these cases, it is better to detect eye blinks through RGB video footage. This however, becomes increasingly difficult with the variation in environmental factors such as camera angle, lighting, and head movement. Our proposed model, called Dual Embedding Video Vision Transformer (DE-ViViT), uses modified tubelet embedding and residual embeddings to mitigate these difficulties, allowing the model to retain high accuracy across changes in these factors.

For more detailed information about the model architecture and training process, please visit the paper link or the official implementation repository:
(paper link)
https://openaccess.thecvf.com/content/WACV2024/html/Hong_Robust_Eye_Blink_Detection_Using_Dual_Embedding_Video_Vision_Transformer_WACV_2024_paper.html

(official implementation repository link)
https://github.com/hongtuna/Eyeblink-detection-using-ViViT/tree/main

## Usage

Install the packages in requirements.txt and run the code “de-vivit_webcam_demo.py”. This accesses the webcam on your device and displays the timepoint of eye blinks and the total number of eye blinks within the session.

If you want to detect blinks from a recorded video by iterating through frames, try running the code “blink_demo_recordedvid.ipynb”. This accesses a pre-recorded video and displays the frames and blinks through opencv, iterating through the frames one by one by pressing any key except ‘q’.

The eye regions are detected with the yolov8 face model [(link)](https://github.com/akanametov/yolov8-face).

## Demo Video
The following demo videos demonstrate the DE-ViViT model’s ability to adapt to changes in lighting, camera angle, and head movement.

