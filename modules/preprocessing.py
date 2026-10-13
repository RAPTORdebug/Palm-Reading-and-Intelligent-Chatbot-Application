import cv2 as cv
import numpy as np
from modules.validation import validation

def preprocessing(left_palm, right_palm):
    val, left, right, left_gray, right_gray = validation(
        left_palm, right_palm
    )

    completed = False

    if val:
        print("Validation passed - started preprocessing")
        # gray
        left_gray = left_gray
        right_gray = right_gray
        # blur
        left_blur = cv.GaussianBlur(left_gray, (5, 5), 0)
        right_blur = cv.GaussianBlur(right_gray, (5, 5), 0)
        # contrast enhancement 
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8),)
        left_enhanced = clahe.apply(left_blur)
        right_enhanced = clahe.apply(right_blur)
        # binary segmentation
        _, left_binary = cv.threshold(left_enhanced,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU,)
        _, right_binary = cv.threshold(right_enhanced,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU,)
        # reducing noise
        kernal = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        left_cleaned = cv.morphologyEx(
            left_binary, cv.MORPH_OPEN, kernal
        )
        right_cleaned = cv.morphologyEx(
                    right_binary, cv.MORPH_OPEN, kernal
        )
        # edge detection - canny(much suitable here)
        left_edges = cv.Canny(left_enhanced, threshold1=50, threshold2=150)
        right_edges = cv.Canny(right_enhanced, threshold1=50, threshold2=150)

        results = {
            "original": (left, right),
            "grayscale": (left_gray, right_gray),
            "blurred": (left_blur, right_blur),
            "enhanced": (left_enhanced, right_enhanced),
            "binary": (left_binary, right_binary),
            "cleaned": (left_cleaned, right_cleaned),
            "edges": (left_edges, right_edges)
        }
        completed = True
        return completed, results
    else:
        print("Validation failed")
        completed = False
        return None, None, None, None