import os
from fastsam import FastSAM, FastSAMPrompt
import cv2
import numpy as np


model = FastSAM('./weights/FastSAM-x.pt')
IMAGE_DIR = './images'
DEVICE = 'cpu'

# List all files in the directory
files = os.listdir(IMAGE_DIR)

# Process each file
for file in files:
    # Only process .png images
    if not file.endswith('.png'):
        continue

    IMAGE_PATH = os.path.join(IMAGE_DIR, file)

    everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.9, iou=0.9,)
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

    # everything prompt
    ann = prompt_process.everything_prompt()

    # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
    # ann = prompt_process.box_prompt(bbox=[[200, 200, 300, 300]])

    # text prompt
    #ann = prompt_process.text_prompt(text='a box')

    # point prompt
    # points default [[0,0]] [[x1,y1],[x2,y2]]
    # point_label default [0] [1,0] 0:background, 1:foreground
    # ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])

    output_path = './output' #os.path.join('./output', file)
    prompt_process.plot(annotations=ann, output=output_path)
