import streamlit as st
import cv2, tempfile
from tensorflow.keras.models import load_model

st.title("Video Action & Tamper Detection")

video = st.file_uploader("Upload video")

if video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    st.video(tfile.name)
