import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
from src.simple_pipeline import SimpleHelmetPipeline
from PIL import Image

st.set_page_config(page_title="Helmet Safety Monitor", layout="wide")
st.title("Professional Helmet Safety Monitor")

@st.cache_resource
def load_ai():
    return SimpleHelmetPipeline(
        yolo_weights="models/weights/yolo_best.pt",
        classifier_weights="models/weights/helmet_classifier_best.pt",
        device="cpu"
    )

ai_engine = load_ai()
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    with st.spinner('Checking Helmet Compliance...'):
        result_img, stats = ai_engine.process_image(frame)
        
    col_img, col_info = st.columns([3, 1])

    with col_img:
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), width='stretch')

    with col_info:
        st.header("Results Summary")
        st.metric("Safe Riders", stats["helmet"])
        st.metric("Violations", stats["no_helmet"])
        
        st.subheader("Report")
        if stats["details"]:
            df = pd.DataFrame(stats["details"])
            # Only display Rider and Status
            st.table(df[['Rider', 'Status']])

            # CSV Download
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Compliance Report",
                data=csv_data,
                file_name=f"Helmet_Report_{datetime.datetime.now().strftime('%H%M%S')}.csv",
                mime='text/csv',
            )
        else:
            st.info("No riders detected in image.")