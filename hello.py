# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:43:39 2023

@author: Asus
"""
import streamlit as st
import PIL
import torch
from ultralytics import YOLO

# Setting page layout
st.set_page_config(
    page_title="Object Detection",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
    
)
# Creating sidebar
with st.sidebar:
    st.header("Image Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))


# Creating main page heading
st.title("Object Detection")

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )


try:
    # Load the saved model
    model = YOLO('yolov8_saved.pt')
except Exception as ex:
    st.error("Unable to load model.")
    st.error(ex)
    

if st.sidebar.button('Detect Objects'):
    # Create a list of the labels
   labels = ['C 48 2.5', 'C 48 3.2', 'C 60 3.2', 'C 89 3.6', 'R 20 40 1.2', 'R 20 40 2.5', 'R 60 40 2', 'R 80 40 2', 'R 80 40 2.5', 'R 80 40 2.9', 'R 96 48 1.6', 'R 96 48 2.5', 'S 20 20 1.2', 'S 25 25 1.9', 'S 38 38 2', 'S 38 38 2.5', 'S 50 50 2', 'S 60 60 1.6', 'S 70 70 2', 'S 72 72 1.6', 'S 72 72 2']

  # Add the labels argument to the plot() method
   res = model.predict(uploaded_image, classes=labels)
   res_plotted = res[0].plot(labels=labels)[:, :, ::-1]
   boxes = res[0].boxes
   
   with col2:
        st.image(res_plotted,
                 caption='Detected Image',
                 use_column_width=True
                 )
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")

