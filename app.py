import streamlit as st
from io import BytesIO
from modules.validate_image import validate_image
from modules.preprocessing import preprocessing_pip
from modules.feature_detection import feature_dect
st.set_page_config(
    page_title="AI Palm Reading",
    layout="wide",
)

st.title("Palm Reading")
st.write(
    "Upload an image or use your camera to capture each palm. "
    "Both images will be validated before image processing."
)

defaults = {
    "left_image": None,
    "right_image": None,
    "left_source_bytes": None,
    "right_source_bytes": None,
    "left_valid": False,
    "right_valid": False,
    "left_upload_key": 0,
    "right_upload_key": 0,
    "left_camera_key": 0,
    "right_camera_key": 0,
    "left_method": "Upload Image",
    "right_method": "Upload Image",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def remove_left_image():
    st.session_state.left_image = None
    st.session_state.left_source_bytes = None
    st.session_state.left_valid = False

    st.session_state.left_upload_key += 1
    st.session_state.left_camera_key += 1

def remove_right_image():
    st.session_state.right_image = None
    st.session_state.right_source_bytes = None
    st.session_state.right_valid = False

    st.session_state.right_upload_key += 1
    st.session_state.right_camera_key += 1


def change_left_method():
      remove_left_image()
  
def change_right_method():
      remove_right_image()


# image validation 
def process_image(image, side):

    if image is None:
        return

    try:
        image_bytes = image.getvalue()

        # Avoid validating the same image on every rerun
        if (
            image_bytes
            == st.session_state[f"{side}_source_bytes"]
        ):
            return

        is_valid = validate_image(
            BytesIO(image_bytes)
        )

        st.session_state[f"{side}_source_bytes"] = image_bytes

        if is_valid:
            st.session_state[f"{side}_image"] = image_bytes
            st.session_state[f"{side}_valid"] = True
            # st.success(f"{side.title()} palm: {message}")

        else:
            st.session_state[f"{side}_image"] = None
            st.session_state[f"{side}_valid"] = False
            st.error(f"{side.title()} palm:Invalid image")

    except Exception as e:
        st.session_state[f"{side}_image"] = None
        st.session_state[f"{side}_valid"] = False
        st.session_state[f"{side}_source_bytes"] = None

        st.error(f"Error processing {side} palm: {e}")
        st.exception(e)

def palm_input_section(side, title):
    """
    Display an upload/camera selector and image preview
    for the specified palm.
    """

    st.header(title)

    method = st.radio(
        f"Choose an option for your {side} palm:",
        ["Upload Image", "Open Camera"],
        key=f"{side}_method",
        horizontal=True,
        on_change=(
            change_left_method
            if side == "left"
            else change_right_method
        ),
    )

    if method == "Upload Image":
        uploaded_image = st.file_uploader(
            f"Upload {title} (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key=(
                f"{side}_upload_"
                f"{st.session_state[f'{side}_upload_key']}"
            ),
        )

        if uploaded_image is not None:
            process_image(uploaded_image, side)

    else:
        camera_image = st.camera_input(
            f"Take a picture of your {side} palm",
            key=(
                f"{side}_camera_"
                f"{st.session_state[f'{side}_camera_key']}"
            ),
        )

        if camera_image is not None:
            process_image(camera_image, side)

    saved_image = st.session_state[f"{side}_image"]

    if saved_image is not None:
        col1, col2 = st.columns([10, 1])

        with col1:
            st.image(
                saved_image,
                caption=f"{title} — validated image",
                width=500,
            )

        with col2:
            st.write("")
            st.button(
                "X",
                key=f"remove_{side}",
                help=f"Remove {side} palm image",
                on_click=(
                    remove_left_image
                    if side == "left"
                    else remove_right_image
                ),
            )

left_col, right_col = st.columns(2, gap="large")

with left_col:
    palm_input_section("left", "Left Palm")

with right_col:
    palm_input_section("right", "Right Palm")

st.divider()

left_ready = (
    st.session_state.left_valid
    and st.session_state.left_image is not None
)

right_ready = (
    st.session_state.right_valid
    and st.session_state.right_image is not None
)

both_ready = left_ready and right_ready

if both_ready:
    st.success(
        "Both palm images are ready for image processing!"
    )

elif left_ready or right_ready:
    st.info(
        "One valid palm image is ready. "
        "Please add the other palm."
    )

else:
    st.info(
        "Upload or capture valid images of both palms "
        "to continue."
    )

# preprocessing and feature dect
def display_preprocessing_results(side, image_bytes):
    """
    Preprocess the palm image, detect features, and display
    """
    #Preprocessing
    success, result = preprocessing_pip(BytesIO(image_bytes))

    if not success:
        st.error(
            f"Could not preprocess the {side} palm: {result}"
        )
        return

    #Feature detection
    feature_success, feature_result = feature_dect(
        result["resized"]
    )

    if not feature_success:
        st.error(
            f"Feature detection failed for the {side} palm: "
            f"{feature_result}"
        )
        return

    # Combine preprocessing and feature detection results
    result.update(feature_result)

    st.subheader(
        f"{side.title()} Palm — Image Processing Results"
    )

    images = [
        ("Original Image", result["original"]),
        ("Resized Image", result["resized"]),
        ("Grayscale Image", result["gray"]),
        ("Gaussian Smoothing", result["blurred"]),
        ("Contrast Enhancement", result["enhanced"]),
        ("Otsu Thresholding", result["binary"]),
        ("Morphological Opening", result["cleaned"]),
        ("Canny Edge Detection", result["edges"]),

        ("Palm Mask", result["palm_mask"]),
        ("Palm Edges", result["palm_edges"]),
        ("Blackhat Crease Enhancement", result["blackhat"]),
        ("Crease Mask", result["crease_mask"]),
        ("Combined Edges and Creases", result["combined"]),
        ("Feature Detection", result["output"])
    ]

    
    for start in range(0, len(images), 3):
        cols = st.columns(3)

        for col_idx, col in enumerate(cols):
            index = start + col_idx

            if index >= len(images):
                break

            image_title, image_data = images[index]

            with col:
                st.image(
                    image_data,
                    caption=image_title,
                    use_container_width=True,
                )

    #detected-line count
    if "line_count" in result:
        st.info(
            f"Detected line segments: {result['line_count']}"
        )
    st.success(
        f"{side.title()} palm processing completed."
    )

if both_ready:

    st.header("Digital Image Processing")

    st.write(
        "Click the button below to apply the image-processing ")

    if st.button(
        "Process Both Palm Images",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner("Processing both palm images..."):

            display_preprocessing_results(
                "left",
                st.session_state.left_image,
            )

            st.divider()

            display_preprocessing_results(
                "right",
                st.session_state.right_image,
            )

        st.success(
            "Image-processing stage finished.")