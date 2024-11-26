# Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Set the page configuration
st.set_page_config(
    page_title="🌟 YOLOv8 Detection",
    page_icon="🪐",
)

# Page Title
st.title("🌟 YOLOv8 Detection")

# Description
st.write(
    "This page allows you to upload an image and use a pretrained YOLOv8 model to detect craters and boulders. "
    "Once an image is uploaded, the model will make predictions and visualize the results."
)

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# If an image is uploaded
if uploaded_file is not None:
    # Load the image
    img = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Initialize the model
    model = YOLO("models/yolov8.pt")

    # Make predictions
    results = model(img)

    # Function to visualize the predictions
    def visualize_detections(image, results):
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100))

        # Display the image
        ax.imshow(image)
        ax.axis("off")

        # Draw bounding boxes on the image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = box.xyxy[0].numpy()
                conf = box.conf[0].item()

                # Define color and label based on confidence
                label = "Boulder" if conf <= 0.3 else "Crater"
                color = "blue" if conf <= 0.3 else "red"

                # Draw the rectangle for the bounding box
                rect = plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=2,
                )
                ax.add_patch(rect)

                # Add label with confidence score
                ax.text(
                    x_min,
                    y_min - 5,
                    f"{label} {conf:.2f}",
                    bbox=dict(facecolor="yellow", alpha=0.5),
                    fontsize=8,
                    color="black",
                )

        # Adjust layout to remove padding
        plt.tight_layout()

        # Show the plot using streamlit
        st.pyplot(fig)

    # Visualize detections
    st.write("Detection Results:")
    visualize_detections(img, results)

else:
    st.warning("⚠️ Please upload an image to get started.")
