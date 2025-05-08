import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
from PIL import Image

# Function to resize image while maintaining aspect ratio
def resize_image(image, width=800):
    height, original_width = image.shape[:2]
    aspect_ratio = width / original_width
    new_height = int(height * aspect_ratio)
    return cv2.resize(image, (width, new_height))

# Function to process images and detect faces
def detect_faces_in_images(reference_image, folder_path, threshold=0.5):
    # Load and process the reference image
    imgRef = face_recognition.load_image_file(reference_image)
    imgRef = cv2.cvtColor(imgRef, cv2.COLOR_BGR2RGB)  # Convert to RGB
    encodeRef = face_recognition.face_encodings(imgRef)[0]

    # Initialize a list to store results
    results = []

    # Loop through each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            imgTest = face_recognition.load_image_file(image_path)
            imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)  # Convert to RGB

            # Find all face locations and encodings in the test image
            faceLocations = face_recognition.face_locations(imgTest)
            faceEncodings = face_recognition.face_encodings(imgTest, faceLocations)

            # Loop through each detected face in the test image
            for faceLoc, encodeFace in zip(faceLocations, faceEncodings):
                faceDist = face_recognition.face_distance([encodeRef], encodeFace)
                
                # If the face distance is below the threshold, consider it a match
                if faceDist[0] < threshold:
                    cv2.rectangle(imgTest, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)
                    cv2.putText(imgTest, f'Match {round(faceDist[0], 2)}', (faceLoc[3], faceLoc[0] - 10), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
                    results.append((filename, imgTest))

    return results

# Streamlit app
st.title("Face Detection App")

# Upload reference image
reference_image = st.file_uploader("Upload Reference Image", type=["jpg", "jpeg", "png"])

# Upload folder containing multiple images
folder_path = st.text_input("Enter the path to the folder containing images:")

# Set threshold for face similarity
threshold = st.slider("Set threshold for face similarity", 0.0, 1.0, 0.5)

if reference_image and folder_path:
    # Save the uploaded reference image to a temporary file
    with open("temp_ref.jpg", "wb") as f:
        f.write(reference_image.getbuffer())

    # Detect faces in the images
    results = detect_faces_in_images("temp_ref.jpg", folder_path, threshold)

    # Display the results
    for filename, imgTest in results:
        st.image(imgTest, caption=f"Match found in {filename}", use_column_width=True)
else:
    st.write("Please upload a reference image and provide the folder path.")