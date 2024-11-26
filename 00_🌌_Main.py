import cv2
import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="🌌 Martian/Lunar Crater & Boulder Detection",
    page_icon="🌙",
)

# Home page content
st.title("🌌 Martian/Lunar Crater and Boulder Detection")
st.write(
    "This web app is dedicated to exploring our model's ability to detect craters & boulders on Martian and Lunar surfaces, "
    "assisting in identifying significant geographical features that could aid in space exploration."
)

st.header("📊 Dataset Information")

# About the Dataset
st.subheader("🔍 About the Dataset")
st.write(
    "**Purpose**: Efficient detection of craters & boulders is crucial for various space exploration missions. "
    "Current methods still face limitations in robustness and versatility. Modern deep-learning-based object detection methods "
    "are proving to be highly promising; however, accessible, high-quality data for training remains scarce. This dataset was curated to address this gap."
)

# What's in the Dataset
st.subheader("📦 What's in the Dataset")
st.markdown(
    """
    - **Image Data**: Mars and Moon surface images containing craters and boulders, sourced from various institutions.
        - **Mars Images**: Mainly from Arizona State University (ASU) and United States Geological Survey (USGS).
        - **Moon Images**: All from the NASA Lunar Reconnaissance Orbiter mission.
        - **Preprocessing**: All images are resized to 640x640 and normalized using RoboFlow.
    - **Labels**: Each image has a corresponding label file in YOLOv8 text format, annotated for crater and boulder detection.
    - **Trained YOLOv8 Model**: Pretrained YOLOv8 models are included. This model helps in crater and boulder detection with accuracy.
"""
)

# Dataset Challenges
st.subheader("⚠️ Dataset Challenges")
st.write(
    "- **Variable Crater and Boulder Sizes**: Craters and Boulders vary significantly in size.\n"
    "- **Diverse Surfaces**: Combining Martian and Lunar surfaces means varied crater shapes, textures, and colors.\n"
    "- **Limited Image Count**: Currently, the dataset contains around 100 images for training, though we plan to add more over time."
)

# Performance Metrics
st.subheader("📈 Training Results")
st.write(
    "In training using the YOLOv8 framework, we achieved the following performance metrics:"
)
st.markdown(
    """
    - **mAP@50**: 0.691
    - **Precision**: 0.608
    - **Recall**: 0.659
    - **mAP@50-95**: 0.381
    """
)
st.write("Training results were saved to `runs/detect/train2`.")

# Sample Images and Detection Results section
st.header("📸 Sample Images and Detection Results")
st.write(
    "Below are sample images from our dataset along with prediction outputs from our trained model:"
)


# Get the path to image
image_path = "./runs/detect/train2/val_batch0_pred.jpg"

# Load the image
image = cv2.imread(image_path)

# If image is found
if image is not None:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="🌌 Martian/Lunar Detection", use_column_width=True)
else:
    st.error("🚨 Image not found. Please check the path.")

# Model Training Section
st.header("🛠️ Model Training")
st.write("Our model is trained using YOLOv8 (Ultralytics 8.3.25).")

st.subheader("📐 Mode Training Google Colab Notebook")
st.link_button(
    label="YOLOv8 Model Training Notebook",
    url="https://colab.research.google.com/drive/1TBVIMGz75oiO4-53HCKT1ksaMytRZx5q"
)

st.subheader("🔧 Training Configuration")
st.code(
    """
# Train the model
model.train(
    data="data.yaml", epochs=100, plots=True,
    val=True, dropout=0.05, verbose=True,
    batch=16, rect=True, device=0
)
""",
    language="python",
)
