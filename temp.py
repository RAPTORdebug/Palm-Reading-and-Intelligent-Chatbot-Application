
import cv2 as cv
from modules.feature_detection import feature_detection

left_palm = "./modules/left-mine.jpeg"
right_palm = "./modules/right-mine.jpeg"

val, results = feature_detection(left_palm, right_palm)

if val:
    print("Feature detection completed")

    # Display all preprocessing results
    for stage, (left_img, right_img) in results["preprocessing"].items():
        cv.imshow(f"Preprocessing - Left {stage.title()}", left_img)
        cv.imshow(f"Preprocessing - Right {stage.title()}", right_img)

    # Display all feature detection results
    for name, image in results["features"].items():
        # Landmark coordinates are not images
        if name.endswith("_points"):
            continue

        cv.imshow(name.replace("_", " ").title(), image)

    cv.waitKey(0)
    cv.destroyAllWindows()

else:
    print("Feature detection failed:", results)
