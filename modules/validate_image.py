import cv2 as cv 
import numpy as np
import os

# left_palm = 'left-mine.jpeg'
# right_palm = 'right-mine.jpeg'

blur_thershold = 100
min_resolution = (480, 480)
# use 640, 640 later
min_brightness = 50
max_brightness = 205


def validation(left_palm, right_palm):

    left_extension = os.path.splitext(left_palm)[1].lower()
    right_extension = os.path.splitext(right_palm)[1].lower()

    # Default report for invalid images
    validation_report = {
        "resolution": {
            "passed": False,
            "left": "Unable to read",
            "right": "Unable to read",
            "required": "480 x 480"
        },
        "lighting": {
            "passed": False,
            "left": "N/A",
            "right": "N/A",
            "required": "50 - 205"
        },
        "sharpness": {
            "passed": False,
            "left": "N/A",
            "right": "N/A",
            "required": "100 or higher"
        }
    }

    # check file extensions
    allowed = [".jpg", ".jpeg", ".png"]

    if left_extension not in allowed or right_extension not in allowed:
        return False, None, None, None, None, validation_report

    left = cv.imread(left_palm)
    right = cv.imread(right_palm)

    if left is None or right is None:
        return False, left, right, None, None, validation_report

    left_height, left_width = left.shape[:2]
    right_height, right_width = right.shape[:2]

    left_resolution = (
        left_width >= min_resolution[0]
        and left_height >= min_resolution[1]
    )

    right_resolution = (
        right_width >= min_resolution[0]
        and right_height >= min_resolution[1]
    )

    left_resized = cv.resize(left, min_resolution, interpolation=cv.INTER_AREA)

    right_resized = cv.resize(right, min_resolution, interpolation=cv.INTER_AREA)

    left_gray = cv.cvtColor(left_resized, cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(right_resized, cv.COLOR_BGR2GRAY)

    left_brightness = np.mean(left_gray)
    right_brightness = np.mean(right_gray)

    left_lighting = min_brightness <= left_brightness <= max_brightness
    right_lighting = min_brightness <= right_brightness <= max_brightness

    left_blur_score = cv.Laplacian(left_gray, cv.CV_64F).var()

    right_blur_score = cv.Laplacian(right_gray, cv.CV_64F).var()

    left_sharp = left_blur_score >= blur_thershold
    right_sharp = right_blur_score >= blur_thershold

    # create validation report
    validation_report = {
        "resolution": {
            "passed": bool(left_resolution and right_resolution),
            "left": f"{left_width} x {left_height}",
            "right": f"{right_width} x {right_height}",
            "required": "480 x 480"
        },
        "lighting": {
            "passed":  bool(left_lighting and right_lighting),
            "left": round(float(left_brightness), 2),
            "right": round(float(right_brightness), 2),
            "required": "50 - 205"
        },
        "sharpness": {
            "passed": bool(left_sharp and right_sharp),
            "left": round(float(left_blur_score), 2),
            "right": round(float(right_blur_score), 2),
            "required": "100 or higher"
        }
    }

    # final validation
    valid = (
        left_resolution and right_resolution
        and left_lighting and right_lighting
        and left_sharp and right_sharp
    )

    return (
        valid,
        left,
        right,
        left_gray,
        right_gray,
        validation_report
    )



# val, left, right = validation(left_palm, right_palm)
