import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the trained model
model_path = "https://drive.google.com/file/d/1em_4pDcMztrkxm1gH0WWmf9I8brKAzkj/view?usp=drive_link"
model = tf.keras.models.load_model(model_path)

# Define class labels for potato leaf diseases
class_labels = ['Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy']

# Custom CSS for styling
st.markdown(
    """
    <style>
        body, .stApp {
            background-color: #D2B48C !important; /* Earthy brown */
        }
        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .uploadedFile {
            max-width: 400px;
        }
        img {
            max-width: 300px; /* Smaller image size */
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown("<h1>🥔 Potato Leaf Disease Classification</h1>", unsafe_allow_html=True)
st.write("Upload an image of a potato leaf to classify its disease.")

# File uploader with smaller size
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="file_uploader")

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=False)
    
    # Ensure image is in RGB mode
    image = image.convert("RGB")
    
    # Preprocess the image
    image = image.resize((128, 128))  # Resize to match model input size
    image_array = np.array(image)  # Keep raw pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get highest probability class
    confidence = np.max(predictions)  # Get confidence score
    
    # Display prediction results
    st.subheader("Prediction")
    st.write(f"*Predicted Class:* {class_labels[predicted_class]}")
    st.write(f"*Confidence:* {confidence:.2f}")
    
    # Display additional message based on prediction
    if class_labels[predicted_class] == 'Potato___Early_blight':
        st.warning("⚠ This leaf has Early Blight. Consider using fungicides and improving field management.")
    elif class_labels[predicted_class] == 'Potato___Late_blight':
        st.error("🚨 This leaf has Late Blight. Immediate action is needed to prevent crop loss!")
    else:
        st.success("✅ This potato leaf is healthy!")
