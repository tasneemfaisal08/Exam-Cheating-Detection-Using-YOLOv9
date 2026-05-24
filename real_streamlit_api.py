import numpy as np
import pandas as pd
import pickle
from PIL import Image
import streamlit as st 
from streamlit_webrtc import webrtc_streamer
import pickle
import torch 

# Open the pickled model file in binary mode
with open(r'E:/Third year/second term/Aplied ML/project_environment/yolov_model.pkl', 'rb') as file:
    # Load the pickled model
    cheating_Detection = pickle.load(file)


def main():
    st.title("Cheating Detection System")
    st.sidebar.title("Cheating Detection System")
    

    st.sidebar.success("Select The Mode You Want")
    #st.sidebar.s['Real Time Detiction']

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Video")
        st.video("E:\Third year\second term\Aplied ML\project_environment\WhatsApp Video 2024-05-07 at 19.36.47_7b440f12.mp4")
    
    with col2:
        with st.spinner("Detecting..."):
            st.subheader("Your Video after Detection")
            st.video("E:\Third year\second term\Aplied ML\project_environment\WhatsApp Video 2024-05-07 at 19.36.47_7b440f12 (1).mp4")
        
    webrtc_streamer(key='key')



    #start code
#button = st.button('start')

if __name__=='__main__':
    main()

