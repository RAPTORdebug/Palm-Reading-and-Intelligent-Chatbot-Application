import base64
import os
import tempfile
import cv2 as cv
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from modules.feature_detection import feature_detection

app = Flask(__name__, static_folder="static")

def to_data_uri(img):
    """Convert an OpenCV image array into a Base64 image URI."""
    if img is None:
        return None

    if not isinstance(img, np.ndarray):
        return None

    if img.dtype != np.uint8:
        if img.size > 0 and img.max() <= 1:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    ok, buf = cv.imencode(".png", img)

    if not ok:
        return None

    encoded = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def make_json_safe(obj):
    """Convert NumPy values into standard JSON-compatible Python values."""
    if isinstance(obj, dict):
        return {
            key: make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(value) for value in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


@app.get("/")
def index():
    return send_from_directory(
        app.root_path,
        "palm-lab-paper.html"
    )


@app.post("/modules/feature_detection")
def detect():

    left = request.files.get("left")
    right = request.files.get("right")

    if not left or not right:
        return jsonify(
            success=False,
            error="Both images are required."
        ), 400

    with tempfile.TemporaryDirectory() as d:

        lp = os.path.join(d, "left_palm.jpg")
        rp = os.path.join(d, "right_palm.jpg")

        left.save(lp)
        right.save(rp)

        val, results = feature_detection(lp, rp)

    # Return detailed validation errors if validation or detection fails.
    if not val:
        return jsonify(
            success=False,
            error=make_json_safe(results)
        ), 200

    # Convert preprocessing images into Base64 image URIs.
    pre = {
        stage: {
            "left": to_data_uri(l),
            "right": to_data_uri(r)
        }
        for stage, (l, r) in results["preprocessing"].items()
    }

    # Convert feature images into Base64 image URIs.
    # Exclude point-coordinate data from the image conversion.
    feats = {
        name: to_data_uri(img)
        for name, img in results["features"].items()
        if not name.endswith("_points")
    }

    # Return the converted images, NOT the original NumPy arrays.
    return jsonify(
        success=True,
        preprocessing=pre,
        features=feats
    )

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
