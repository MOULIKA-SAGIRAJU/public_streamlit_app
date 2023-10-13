# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:39:09 2023

@author: Asus
"""

import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLOv8

# Set the page layout with two columns
col1, col2 = st.columns(2)

# Display the image upload widget in the first column
with col1:
    uploaded_image = st.file_uploader("Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

# Load the YOLOv8 model with the best weights in the second column
with col2:
    model = YOLOv8('yolov8_saved.pt')

# Detect objects in the uploaded image if it is not None
if uploaded_image is not None:

     # Run the YOLOv8 model on the image 
    detections = model.predict(uploaded_image)

    # Postprocess the detections
    detections = detections[detections[:, 5] > 0.5]
    detections = detections[:, :6]

    # Draw bounding boxes and class labels on the image
    for detection in detections:
        bounding_box = detection[:4]
        class_label = int(detection[4])
        confidence = detection[5]

        # Draw a bounding box around the detected object
        cv2.rectangle(uploaded_image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]))

        # Draw the class label of the detected object
        cv2.putText(uploaded_image, str(class_label), (bounding_box[0], bounding_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX)

    # Convert the NumPy array to a PIL Image object
    image = Image.fromarray(cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB))

    # Display the output image in the second column
    st.image(image, caption="Output image")