import cv2 
import numpy as np
from PIL import Image
import streamlit as st


def validate_image(in_image):
    if in_image == None:
        return False, 'Pls upload the image'

    try:
        # no saving directly from the memory
        image = Image.open(in_image).convert('RGB')
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        if width < 250 or height < 250:
            return False, 'Image is too small'
        # grayscale
        gray = cv2.cvtColor(
            image_array, cv2.COLOR_RGB2GRAY
        )
        brightness = np.mean(gray)
        if brightness < 35:
            return False, 'Image is too dark'
        if brightness > 240:
            return False, 'Image is too bright'
        #contrast
        contrast = np.std(gray)
        if contrast < 15:
            return False, "Image has insufficient contrast. Try again."

        #blur
        blur_score = cv2.Laplacian(
            gray, cv2.CV_64F
        ).var()
        if blur_score < 40:
            return False, "Image looks blurry"
        return True

    except Exception:
        return False, "Unable to read this image. Please upload a JPG or PNG."
