
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
    valid = False
    left_extension = os.path.splitext(left_palm)[1].lower()
    right_extension = os.path.splitext(right_palm)[1].lower()

    if left_extension in [".jpg", ".jpeg", ".png"] and right_extension in [".jpg", ".jpeg", ".png"]:
        left_palm = cv.imread(left_palm)
        right_palm = cv.imread(right_palm)

        if left_palm is not None and right_palm is not None:
            # check original image resolution
            left_height, left_width = left_palm.shape[:2]
            right_height, right_width = right_palm.shape[:2]

            left_resolution = left_width >= min_resolution[0] and left_height >= min_resolution[1]
            right_resolution = right_width >= min_resolution[0] and right_height >= min_resolution[1]

            if left_resolution and right_resolution:
                print("Both images have sufficient resolution")
            else:
                print("One or both images have low resolution")
                print("Left resolution:", left_width, "x", left_height)
                print("Right resolution:", right_width, "x", right_height)

            # resize images for processing(this is for now will make a new pip)
            left_resized = cv.resize(left_palm, (min_resolution), interpolation=cv.INTER_AREA)
            right_resized = cv.resize(right_palm, (min_resolution), interpolation=cv.INTER_AREA)

            left_gray = cv.cvtColor(left_resized, cv.COLOR_BGR2GRAY)
            right_gray = cv.cvtColor(right_resized, cv.COLOR_BGR2GRAY)

            left_brightness = np.mean(left_gray)
            right_brightness = np.mean(right_gray)

            left_lighting = min_brightness <= left_brightness <= max_brightness
            right_lighting = min_brightness <= right_brightness <= max_brightness

            if left_lighting and right_lighting:
                print("Both images have acceptable lighting")
            else:
                print("One or both images have poor lighting")
                print("Left brightness:", left_brightness)
                print("Right brightness:", right_brightness)

            left_blur_score = cv.Laplacian(left_gray, cv.CV_64F).var()
            right_blur_score = cv.Laplacian(right_gray, cv.CV_64F).var()

            left_sharp = left_blur_score >= blur_thershold
            right_sharp = right_blur_score >= blur_thershold

            if left_sharp and right_sharp:
                print("Both images are sufficiently sharp")
            else:
                print("One or both images are too blurry")
                print("Left sharpness:", left_blur_score)
                print("Right sharpness:", right_blur_score)

            #display images only if all checks pass
            if (left_resolution and right_resolution and
                left_lighting and right_lighting and
                left_sharp and right_sharp):

                print("All validation checks passed")
                # cv.imshow("Left", left_palm)
                # cv.imshow("Right", right_palm)
                # cv.imshow("Left Grayscale", left_gray)
                # cv.imshow("Right Grayscale", right_gray)

                valid = True
            else:
                print("Image validation failed")
                valid = False

        else:
            print("Unable to read the image")
            valid = False
    else:
        print("Only JPG, JPEG, and PNG files are allowed")
        valid = False

    cv.waitKey(0)
    cv.destroyAllWindows()
    return valid, left_palm, right_palm, left_gray, right_gray


# val, left, right = validation(left_palm, right_palm)
