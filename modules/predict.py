
import cv2 as cv
from pathlib import Path
from ultralytics import YOLO

# Load YOLO Pose model
MODEL_PATH = Path(__file__).resolve().parent.parent / "best.pt"
model = YOLO(str(MODEL_PATH))

classes = ['fate', 'head', 'heart', 'life']

colors = {
    'fate': (255, 0, 0),
    'head': (0, 255, 0),
    'heart': (0, 0, 255),
    'life': (0, 255, 255)
}


def predict(image):

    if image is None:
        return None

    # Work with the already-preprocessed image
    annotated = image.copy()

    results = model.predict(
        source=annotated,
        conf=0.5,
        verbose=False
    )

    for result in results:

        if result.keypoints is None or result.boxes is None:
            continue

        if result.keypoints.conf is None:
            continue

        keypoints = result.keypoints.xy.cpu().numpy()
        kp_conf = result.keypoints.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i, points in enumerate(keypoints):

            if i >= len(class_ids):
                continue

            class_id = class_ids[i]

            if not 0 <= class_id < len(classes):
                continue

            class_name = classes[class_id]
            color = colors[class_name]

            # Draw confident keypoints
            for j, point in enumerate(points):

                x, y = map(int, point)

                if kp_conf[i][j] > 0.5:
                    cv.circle(
                        annotated,
                        (x, y),
                        4,
                        color,
                        -1
                    )

            # Connect consecutive confident keypoints
            for j in range(len(points) - 1):

                if (
                    kp_conf[i][j] > 0.5
                    and kp_conf[i][j + 1] > 0.5
                ):

                    x1, y1 = map(int, points[j])
                    x2, y2 = map(int, points[j + 1])

                    cv.line(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        color,
                        3
                    )

            # Add class label
            valid_points = points[kp_conf[i] > 0.5]

            if len(valid_points) > 0:

                label_x, label_y = map(int, valid_points[0])

                cv.putText(
                    annotated,
                    class_name.upper(),
                    (label_x, max(20, label_y - 12)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

    return annotated
