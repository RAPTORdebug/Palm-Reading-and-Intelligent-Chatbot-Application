import streamlit as st
from modules.validate_image import validate_image

st.set_page_config(page_title="Palm Reading")

st.title("Palm Reading")
st.write("Upload an image or use your camera to capture each palm.")

# Initialize session state
defaults = {
    "left_image": None,
    "right_image": None,
    "left_upload_key": 0,
    "right_upload_key": 0,
    "left_camera_key": 0,
    "right_camera_key": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def remove_left_image():
    st.session_state.left_image = None
    st.session_state.left_upload_key += 1
    st.session_state.left_camera_key += 1

def remove_right_image():
    st.session_state.right_image = None
    st.session_state.right_upload_key += 1
    st.session_state.right_camera_key += 1


# Validate image without saving it to disk
def process_image(image, side):
    if image is None:
        return

    try:
        image.seek(0)
        is_valid, message = validate_image(image)
        image.seek(0)

        if is_valid:
            st.session_state[f"{side}_image"] = image.getvalue()
            st.session_state[f"{side}_valid"] = True
            st.success(message)
        else:
            st.session_state[f"{side}_image"] = None
            st.session_state[f"{side}_valid"] = False
            st.error(message)

    except Exception as error:
        st.session_state[f"{side}_image"] = None
        st.session_state[f"{side}_valid"] = False
        st.error(f"Unable to process image: {error}")


# Initialize validation states
for key in ["left_valid", "right_valid"]:
    if key not in st.session_state:
        st.session_state[key] = False


#left
st.header("Left Palm")

left_method = st.radio(
    "Choose an option for your left palm:",
    ["Upload Image", "Open Camera"],
    key="left_method",
    horizontal=True,
)

if left_method == "Upload Image":
    left_upload = st.file_uploader(
        "Upload Left Palm (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key=f"left_upload_{st.session_state.left_upload_key}",
    )

    if left_upload is not None:
        process_image(left_upload, "left")

else:
    left_camera = st.camera_input(
        "Take a picture of your left palm",
        key=f"left_camera_{st.session_state.left_camera_key}",
    )

    if left_camera is not None:
        process_image(left_camera, "left")


if st.session_state.left_image is not None:
    col1, col2 = st.columns([10, 1])

    with col1:
        st.image(
            st.session_state.left_image,
            caption="Left Palm",
            use_container_width=True,
        )

    with col2:
        st.button(
            "❌",
            key="remove_left",
            help="Remove left palm image",
            on_click=remove_left_image,
        )

st.divider()


# right palm
st.header("Right Palm")

right_method = st.radio(
    "Choose an option for your right palm:",
    ["Upload Image", "Open Camera"],
    key="right_method",
    horizontal=True,
)

if right_method == "Upload Image":
    right_upload = st.file_uploader(
        "Upload Right Palm (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key=f"right_upload_{st.session_state.right_upload_key}",
    )

    if right_upload is not None:
        process_image(right_upload, "right")

else:
    right_camera = st.camera_input(
        "Take a picture of your right palm",
        key=f"right_camera_{st.session_state.right_camera_key}",
    )

    if right_camera is not None:
        process_image(right_camera, "right")


if st.session_state.right_image is not None:
    col1, col2 = st.columns([10, 1])

    with col1:
        st.image(
            st.session_state.right_image,
            caption="Right Palm",
            use_container_width=True,
        )

    with col2:
        st.button(
            "❌",
            key="remove_right",
            help="Remove right palm image",
            on_click=remove_right_image,
        )

st.divider()

if (
    st.session_state.left_valid
    and st.session_state.right_valid
    and st.session_state.left_image is not None
    and st.session_state.right_image is not None
):
    st.success("Both palm images are ready for processing!")

elif (
    st.session_state.left_image is not None
    or st.session_state.right_image is not None
):
    st.info("One valid palm image is ready. Please add the other palm.")

else:
    st.info("Upload or capture valid images of both palms to continue.")