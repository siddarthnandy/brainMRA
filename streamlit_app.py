import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('aneurysm_detection_model.h5')

st.title("🧐 Brain Aneurysm AI Detection App")

st.write(
    "This application uses an AI model to predict the presence of brain aneurysms in MRI slices. "
    "You can upload a set of MRI images and see the predictions of our AI model."
)

# File uploader for images
uploaded_files = st.file_uploader("Upload MRI slices", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.write("## Uploaded Images")
    
    data = []  # List to store prediction results
    
    for uploaded_file in uploaded_files:
        # Load the image
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(64, 64), color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Get prediction from the model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        probability = np.max(prediction)
        
        # Determine the predicted label
        label = "Aneurysm Detected" if predicted_class == 1 else "No Aneurysm"
        
        # Append to data list
        data.append({
            "Image Name": uploaded_file.name,
            "Prediction": label,
            "Probability": f"{probability:.2f}"
        })
        
        # Display the image and prediction
        st.image(uploaded_file, caption=f"Prediction: {label} (Probability: {probability:.2f})", use_column_width=True)

    # Display results in a table
    st.write("## Summary of Predictions")
    df = pd.DataFrame(data)
    st.write(df)

    # Visualize annotation progress
    st.divider()
    st.write("### Annotation Summary")
    aneurysm_count = len(df[df["Prediction"] == "Aneurysm Detected"])
    total_count = len(df)
    aneurysm_percentage = f"{(aneurysm_count / total_count) * 100:.2f}%"
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Number of Images with Aneurysms", aneurysm_count)
    with col2:
        st.metric("Aneurysm Detection Rate", aneurysm_percentage)

    # Corrected bar chart
    st.bar_chart(df["Prediction"].value_counts())

st.write(
    "Thank you for using the Brain Aneurysm AI Detection App. If you have any concerns, please consult a medical professional."
)
