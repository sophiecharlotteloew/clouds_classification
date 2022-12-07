import streamlit as st
import pandas as pd
from PIL import Image
import uvicorn
import requests
import io
from streamlit_cropper import st_cropper
import numpy as np


url = "https://clouds-ivl76q6s4a-ew.a.run.app/predict"

# Just design settings:
st.set_page_config(
page_title="Quick reference",
page_icon="☁️",
layout="centered")

CSS = """h1 {color: white;}
.stApp {
    background-image: url(https://images.pexels.com/photos/557782/pexels-photo-557782.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2);
    background-size: cover;}"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.sidebar.markdown(f"""# Please upload a photo of the sky or take a photo directly:""")

# Upload file with browsing or photo
uploaded_file = st.sidebar.file_uploader("Upload a JPG file", type="jpg")
uploaded_file_2 = st.sidebar.camera_input("Take a picture")

st.write("### Let's analyze the Sky!")
st.set_option('deprecation.showfileUploaderEncoding', False)

if uploaded_file is not None:
    # Crop the uploaded image on the page
    st.write('#### Please select the part of the sky')
    realtime_update = st.checkbox(label="Update in Real Time", value=True)
    img = Image.open(uploaded_file)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color="blue",
                                aspect_ratio=(1, 1))

    # Manipulate cropped image at will
    st.write("Preview")
    _ = cropped_img.thumbnail((224,224))
    st.image(cropped_img)
    cropped_img = cropped_img.convert(mode = "RGB")

    # Add one pixel if shape is not 200
    if cropped_img.size[0] != 224:
        na = np.array(cropped_img)
        x = na[0,:]
        y = np.expand_dims(na[:,0], axis=1)
        new_img = np.append(na, y, axis=1)
        cropped_img = Image.fromarray(new_img)

    # Convert cropped image to bytes
    cropped_img_bytes = cropped_img.tobytes()

    if st.button('Analyze the weather'):
        # Progress bar
        st.spinner("Sending the image to the API ...")
        # Sending to API for prediction
        res = requests.post(url, files={'bytes': cropped_img_bytes})
        res = res.content
        res = res.decode("utf-8")

        st.write('Analysis of weather performed ! ⛅')
        st.write(f'### {res}')

if uploaded_file_2 is not None:
    # Crop the uploaded image on the page
    st.write('#### Please select the part of the sky')
    realtime_update = st.checkbox(label="Update in Real Time", value=True)
    img = Image.open(uploaded_file_2)
    if not realtime_update:
        st.write("Double click to save crop")

    # Get a cropped image from the frontend
    cropped_img_2 = st_cropper(img, realtime_update=realtime_update, box_color="blue",
                                aspect_ratio=(1, 1))

    st.write(cropped_img_2.size)

    # Manipulate cropped image at will
    st.write("Preview")
    _ = cropped_img_2.thumbnail((224,224))
    st.image(cropped_img_2)
    cropped_img_2 = cropped_img_2.convert(mode = "RGB")

    # Add one pixel if shape is not 200
    if cropped_img_2.size[0] != 224:
        na = np.array(cropped_img_2)
        x = na[0,:]
        y = np.expand_dims(na[:,0], axis=1)
        new_img = np.append(na, y, axis=1)
        cropped_img_2 = Image.fromarray(new_img)

    # Convert cropped image to bytes
    cropped_img_bytes = cropped_img_2.tobytes()

    if st.button('Analyze the weather'):
        # Progress bar
        st.spinner("Sending the image to the API ...")
        # Sending to API for prediction
        res = requests.post(url + "/predict", files={'bytes': cropped_img_bytes})
        res = res.content
        res = res.decode("utf-8")

        st.write('Analysis of weather performed ! ⛅')
        st.write(f'### {res}')

if st.button('Or do you wish for snow?'):
    st.snow()
