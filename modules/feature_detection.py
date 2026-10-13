
import cv2 as cv
import numpy as np
import mediapipe as mp

from modules.preprocessing import preprocessing

mp_hands = mp.solutions.hands


def create_palm_mask(points, image_shape):
    """
    Create an approximate palm-region mask from MediaPipe
    wrist and finger-base landmarks.
    """
    height, width = image_shape[:2]

    # MediaPipe landmark indices:
    # 0 = wrist
    # 1 = thumb CMC
    # 2 = thumb MCP
    # 5 = index finger MCP
    # 9 = middle finger MCP
    # 13 = ring finger MCP
    # 17 = pinky MCP

    # Start with the wrist and finger-base landmarks.
    palm_ids = [0, 1, 2, 5, 9, 13, 17]
    palm_points = points[palm_ids].astype(np.float32)

    # Approximate the central palm region.
    hull = cv.convexHull(
        palm_points.astype(np.int32)
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv.fillConvexPoly(mask, hull, 255)

    # Smooth the mask boundary.
    smooth_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (9, 9)
    )

    mask = cv.morphologyEx(
        mask,
        cv.MORPH_CLOSE,
        smooth_kernel
    )

    # Expand slightly to retain creases near the boundary.
    expand_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv.dilate(
        mask,
        expand_kernel,
        iterations=1
    )

    return mask


def remove_small_components(mask, min_area=8):
    """Remove tiny isolated regions from a binary image."""
    count, labels, stats, _ = cv.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)

    for i in range(1, count):
        area = stats[i, cv.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def extract_creases(enhanced, palm_mask):
    """
    Extract dark crease candidates within the approximate
    palm region using black-hat morphology.
    """
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (17, 17)
    )

    blackhat = cv.morphologyEx(
        enhanced, cv.MORPH_BLACKHAT, kernel
    )

    # Restrict detection to the palm region
    blackhat = cv.bitwise_and(
        blackhat, blackhat, mask=palm_mask
    )

    values = blackhat[palm_mask > 0]

    if values.size == 0:
        return None, None

    # Threshold pixels within the palm region
    threshold, _ = cv.threshold(
        values.reshape(-1, 1),
        0,
        255,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    _, crease_mask = cv.threshold(
        blackhat,
        threshold,
        255,
        cv.THRESH_BINARY
    )

    crease_mask = cv.bitwise_and(
        crease_mask, palm_mask
    )

    # Remove isolated noise
    open_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (3, 3)
    )
    crease_mask = cv.morphologyEx(
        crease_mask, cv.MORPH_OPEN, open_kernel
    )

    # Gently connect nearby fragments
    close_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (3, 3)
    )
    crease_mask = cv.morphologyEx(
        crease_mask, cv.MORPH_CLOSE, close_kernel
    )

    # Remove tiny connected components
    crease_mask = remove_small_components(
        crease_mask, min_area=8
    )

    # Ensure no pixels extend outside the palm mask
    crease_mask = cv.bitwise_and(
        crease_mask, palm_mask
    )

    return blackhat, crease_mask


def annotate_palm(original, points, palm_mask, crease_mask):
    """Display the palm mask boundary and crease candidates."""
    height, width = palm_mask.shape[:2]

    # Keep the annotation at the processed image size
    annotated = cv.resize(
        original, (width, height)
    ).copy()

    # Draw the palm-mask boundary
    palm_contours, _ = cv.findContours(
        palm_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    cv.drawContours(
        annotated,
        palm_contours,
        -1,
        (0, 255, 255),
        2
    )

    # Draw crease candidates
    crease_contours, _ = cv.findContours(
        crease_mask,
        cv.RETR_LIST,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in crease_contours:
        if cv.arcLength(contour, False) >= 15:
            cv.drawContours(
                annotated,
                [contour],
                -1,
                (255, 0, 255),
                1
            )

    # Draw MediaPipe landmarks
    for x, y in points:
        cv.circle(
            annotated,
            (int(x), int(y)),
            3,
            (0, 255, 0),
            -1
        )

    # Legend
    cv.putText(
        annotated,
        "Palm outline",
        (8, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv.putText(
        annotated,
        "Crease candidates",
        (8, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )

    return annotated


def feature_detection(left_palm, right_palm):
    """Run preprocessing, detect landmarks, and extract creases."""

    # Run preprocessing once
    comp, results = preprocessing(left_palm, right_palm)

    if not comp:
        return False, "Preprocessing failed"

    # Retrieve preprocessing outputs
    left, right = results["original"]

    left_gray, right_gray = results["grayscale"]
    left_blurred, right_blurred = results["blurred"]
    left_enhanced, right_enhanced = results["enhanced"]
    left_binary, right_binary = results["binary"]
    left_cleaned, right_cleaned = results["cleaned"]
    left_edges, right_edges = results["edges"]

    # Display/detection images are resized to 480x480
    left_display = cv.resize(left, (480, 480))
    right_display = cv.resize(right, (480, 480))

    # MediaPipe expects RGB images
    left_rgb = cv.cvtColor(left_display, cv.COLOR_BGR2RGB)
    right_rgb = cv.cvtColor(right_display, cv.COLOR_BGR2RGB)

    # Detect hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as hands:
        left_hand_results = hands.process(left_rgb)
        right_hand_results = hands.process(right_rgb)

    if (
        not left_hand_results.multi_hand_landmarks
        or not right_hand_results.multi_hand_landmarks
    ):
        return False, "No hand detected in one or both images."

    left_landmarks = (
        left_hand_results.multi_hand_landmarks[0].landmark
    )
    right_landmarks = (
        right_hand_results.multi_hand_landmarks[0].landmark
    )

    # Convert normalized landmarks to processed-image coordinates
    left_height, left_width = left_enhanced.shape[:2]

    left_points = np.array([
        [
            int(np.clip(p.x * left_width, 0, left_width - 1)),
            int(np.clip(p.y * left_height, 0, left_height - 1))
        ]
        for p in left_landmarks
    ], dtype=np.int32)

    right_height, right_width = right_enhanced.shape[:2]

    right_points = np.array([
        [
            int(np.clip(p.x * right_width, 0, right_width - 1)),
            int(np.clip(p.y * right_height, 0, right_height - 1))
        ]
        for p in right_landmarks
    ], dtype=np.int32)

    # Create approximate palm masks
    left_mask = create_palm_mask(
        left_points, left_enhanced.shape
    )
    right_mask = create_palm_mask(
        right_points, right_enhanced.shape
    )

    # Extract crease candidates
    left_blackhat, left_crease_mask = extract_creases(
        left_enhanced, left_mask
    )
    right_blackhat, right_crease_mask = extract_creases(
        right_enhanced, right_mask
    )

    if left_crease_mask is None or right_crease_mask is None:
        return False, "Could not extract one or both palm regions."

    # Create annotated images
    left_annotated = annotate_palm(
        left_display,
        left_points,
        left_mask,
        left_crease_mask
    )
    right_annotated = annotate_palm(
        right_display,
        right_points,
        right_mask,
        right_crease_mask
    )

    # Collect feature results
    feature_results = {
        "left_annotated": left_annotated,
        "right_annotated": right_annotated,

        "left_palm_mask": left_mask,
        "right_palm_mask": right_mask,

        "left_blackhat": left_blackhat,
        "right_blackhat": right_blackhat,

        "left_crease_mask": left_crease_mask,
        "right_crease_mask": right_crease_mask,

        "left_points": left_points,
        "right_points": right_points,

        "left_edges": left_edges,
        "right_edges": right_edges,
    }

    return True, {
        "preprocessing": results,
        "features": feature_results
    }
