from io import StringIO
from pathlib import Path
import streamlit as st
import time
import cv2
import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np
st.header("YOLOV5 COVID-19 PPE IMAGE DETECTION")
file_upload = st.file_uploader("choose file ", type = ["jpg", "png","jpeg"])
model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt")  # load scratch
if file_upload is not None:
    img = Image.open(file_upload)
    img  = model(img)
    st.image(np.squeeze(img.render()))