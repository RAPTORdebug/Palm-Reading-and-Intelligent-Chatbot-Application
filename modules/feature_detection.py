
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

def feature_dect(in_image):
    if in_image is None or in_image.size == 0:
        return False, "Invalid input image."

    original = in_image.copy()

    if original.ndim != 3 or original.shape[2] != 3:
        return False, "Expected a 3-channel RGB image."

    height, width = original.shape[:2]

    # mediaPipe expects RGB
    rgb = original.copy()

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as hands:
        hand_results = hands.process(rgb)

    if not hand_results.multi_hand_landmarks:
        return False, "No hand detected."

    landmarks = hand_results.multi_hand_landmarks[0].landmark

    points = np.array([
        [
            int(np.clip(p.x * width, 0, width - 1)),
            int(np.clip(p.y * height, 0, height - 1))
        ]
        for p in landmarks
    ], dtype=np.int32)

    # Approximate palm region using wrist and finger-base landmarks
    palm_ids = [0, 1, 2, 5, 9, 13, 17]
    palm_points = points[palm_ids]

    hull = cv2.convexHull(palm_points)

    palm_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(palm_mask, hull, 255)

    # Expand ROI moderately
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15, 15)
    )
    palm_mask = cv2.dilate(palm_mask, kernel)

    # Smooth and enhance contrast
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    enhanced = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8)).apply(gray)

    # Suppress small-scale texture and emphasize dark creases
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(enhanced,cv2.MORPH_BLACKHAT,blackhat_kernel)

    # Ignore responses outside the palm
    blackhat = cv2.bitwise_and(blackhat, blackhat, mask=palm_mask)

    # Threshold crease candidates
    roi_values = blackhat[palm_mask > 0]

    if roi_values.size == 0:
        return False, "Could not extract palm region."

    roi_image = roi_values.reshape(-1, 1)
    threshold_value, _ = cv2.threshold(roi_image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    crease_mask = cv2.threshold(blackhat,threshold_value,255,cv2.THRESH_BINARY)[1]

    crease_mask = cv2.bitwise_and(crease_mask, crease_mask, mask=palm_mask)

    # Remove isolated noise
    crease_mask = cv2.morphologyEx(crease_mask,cv2.MORPH_OPEN,np.ones((3, 3), np.uint8))

    # Remove small connected components
    count, labels, stats, _ = cv2.connectedComponentsWithStats(crease_mask, connectivity=8)

    cleaned = np.zeros_like(crease_mask)
    min_area = max(8, int(height * width * 0.00001))

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    # Skeletonization using iterative morphological thinning
    skeleton = np.zeros_like(cleaned)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = cleaned.copy()

    while cv2.countNonZero(work) > 0:
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        residue = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        work = eroded

    # Visualize detected crease pixels
    output = original.copy()
    output[skeleton > 0] = (255, 0, 0)  # RGB blue

    lined = np.zeros_like(original)
    lined[skeleton > 0] = (255, 0, 0)

    return True, {
        "lined": lined,
        "output": output,
        "line_count": int(cv2.countNonZero(skeleton)),
        "palm_mask": palm_mask,
        "palm_edges": cv2.Canny(enhanced, 50, 150),
        "blackhat": blackhat,
        "crease_mask": cleaned,
        "combined": skeleton,
    }
