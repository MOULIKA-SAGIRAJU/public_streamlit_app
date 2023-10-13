# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:48:49 2023

@author: Asus
"""

import streamlit as st
import torch
import YOLOv8
#get input image
image = st.file_uploader("upload an image")
#load model
model = torch.load('yolov8_saved.pt')

#define the function to perform object detection
def detect_objects(image):
    detections = model.predict(image)
    return detections

#function to display objects
def display_objects(detections):
    for detection in detections:
        bbox = detection[0:4]
        confidence = detection[4]
        class_id = detection[5]
#draw the bounding box around the detected image
        st.image(image, bbox, color='red')
        st.write(f'Confidence:{confidence:.3f}')
        st.write(f'classid:{class_id}')

if image is not None:
    detections = detect_objects(image)
    display_objects(detections)
    
