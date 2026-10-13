import streamlit as st
import cv2 as cv
import tempfile
import os

from modules.feature_detection import feature_detection

st.set_page_config(
    page_title="Palm Feature Detection",
    page_icon="🖐️",
    layout="wide"
)

st.title("Palm Feature Detection")

# Upload left and right palm images
col1, col2 = st.columns(2)

with col1:
    left_file = st.file_uploader(
        "Upload Left Palm",
        type=["jpg", "jpeg", "png"],
        key="left"
    )

with col2:
    right_file = st.file_uploader(
        "Upload Right Palm",
        type=["jpg", "jpeg", "png"],
        key="right"
    )

# Display uploaded images
if left_file is not None:
    col1.image(
        left_file,
        caption="Left Palm",
        width=480
    )

if right_file is not None:
    col2.image(
        right_file,
        caption="Right Palm",
        width=480
    )


def display_image_grid(images, columns=3):
    """Display images in a grid with the specified number of columns."""
    cols = st.columns(columns)

    for index, (title, image) in enumerate(images):
        with cols[index % columns]:
            if image is not None:
                if len(image.shape) == 3:
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                st.image(
                    image,
                    caption=title,
                    clamp=True,
                    use_container_width=True
                )


# Run feature detection
if st.button("Run Feature Detection", type="primary"):

    if left_file is None or right_file is None:
        st.warning("Please upload both palm images.")

    else:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:

                left_path = os.path.join(temp_dir, "left_palm.jpg")
                right_path = os.path.join(temp_dir, "right_palm.jpg")

                with open(left_path, "wb") as f:
                    f.write(left_file.getvalue())

                with open(right_path, "wb") as f:
                    f.write(right_file.getvalue())

                with st.spinner("Processing palm images..."):
                    val, results = feature_detection(
                        left_path,
                        right_path
                    )

                if val:
                    st.success("Feature detection completed!")

                    # 1. Preprocessing results
                    st.header("Preprocessing Results")

                    preprocessing_images = []

                    for stage, (left_img, right_img) in (
                        results["preprocessing"].items()
                    ):
                        # Resize only the original images for display
                        if stage.lower() == "original":
                            left_display = cv.resize(left_img, (480, 480))
                            right_display = cv.resize(right_img, (480, 480))
                        else:
                            left_display = left_img
                            right_display = right_img

                        preprocessing_images.append(
                            (f"Left - {stage.title()}", left_display)
                        )
                        preprocessing_images.append(
                            (f"Right - {stage.title()}", right_display)
                        )

                    display_image_grid(preprocessing_images, columns=3)

                    # 2. Feature detection results
                    # Keep annotated images for the final section.
                    st.header("Feature Detection Results")

                    feature_images = []
                    annotated_images = []

                    for name, image in results["features"].items():

                        # Landmark coordinates are not images
                        if name.endswith("_points"):
                            continue

                        title = name.replace("_", " ").title()

                        # Annotated images are displayed last
                        if "annotated" in name.lower():
                            annotated_images.append((title, image))
                        else:
                            feature_images.append((title, image))

                    if feature_images:
                        display_image_grid(feature_images, columns=3)
                    else:
                        st.info("No feature images to display.")

                    # 3. Annotated images at the bottom
                    if annotated_images:
                        st.header("Annotated Images")
                        display_image_grid(
                            annotated_images,
                            columns=3
                        )

                else:
                    st.error("Feature detection failed.")
                    st.write(results)

        except Exception as e:
            st.error(f"An error occurred: {e}")