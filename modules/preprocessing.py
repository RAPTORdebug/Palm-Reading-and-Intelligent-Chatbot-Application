import cv2
import numpy as np
from PIL import Image
from io import BytesIO

def preprocessing_pip(in_image):

    if in_image is None:
        return False, "Plsupload an image."
    
    try:
        #read the image
        if isinstance(in_image, np.ndarray):
            image_array = in_image.copy()
            # OpenCV arrays commonly use BGR; convert to RGB.
            if image_array.ndim == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            if isinstance(in_image, bytes):
                image_source = BytesIO(in_image)
            else:
                image_source = in_image

            if hasattr(image_source, "seek"):
                image_source.seek(0)

            with Image.open(image_source) as image:
                image_array = np.array(image.convert("RGB"))

            if hasattr(image_source, "seek"):
                image_source.seek(0)

        if image_array.size == 0:
            return False, "The uploaded image is empty."

        #Resize proportionally without distorting the palm
        original_height, original_width = image_array.shape[:2]

        max_width = 640
        max_height = 640

        scale = min(max_width / original_width,max_height / original_height,1.0,)
        new_width = max(1, int(original_width * scale))
        new_height = max(1, int(original_height * scale))

        resized = cv2.resize(
            image_array,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR,
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        #noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8),)
        enhanced = clahe.apply(blurred)
        #binary segmentation
        _, binary = cv2.threshold(enhanced,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
        #reduce small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
        )
        #Canny edge detection
        edges = cv2.Canny(
            enhanced,
            threshold1=50,
            threshold2=150,
        )

        # Return every stage for display and further analysis
        results = {
            "original": image_array,
            "resized": resized,
            "gray": gray,
            "blurred": blurred,
            "enhanced": enhanced,
            "binary": binary,
            "cleaned": cleaned,
            "edges": edges,
        }

        return True, results

    except Exception:
        return (
            False,
            "Unable to preprocess this image. "
            "Please check the file and try again.",
        )