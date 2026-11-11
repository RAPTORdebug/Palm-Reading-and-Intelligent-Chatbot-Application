from modules.preprocessing import preprocessing
from modules.predict import predict

def feature_detection(left_palm, right_palm):
    """
    Integrated palm image processing and feature detection.

    Task 02: Image preprocessing
    Task 03: YOLO Pose feature identification
    """

    completed, results = preprocessing(left_palm, right_palm)

    if not completed:
        return False, {
            "message": "Image validation failed.",
            "validation": results["validation"]
        }

    # retrieve resized original images
    left, right = results["original"]

    # detect palm lines using YOLO Pose
    left_annotated = predict(left)
    right_annotated = predict(right)

    if left_annotated is None or right_annotated is None:
        return False, "Palm feature detection failed."

    # collect results
    feature_results = {
        "left_annotated": left_annotated,
        "right_annotated": right_annotated
    }

    return True, {
        "preprocessing": results,
        "features": feature_results
    }
