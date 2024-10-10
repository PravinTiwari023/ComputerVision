import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Orientation Histograms & Descriptors")

# Introduction to Descriptors
st.header("Introduction to Feature Descriptors")
st.write("""
**Feature descriptors** describe the local features of an image in a way that is invariant to changes in scale, rotation, and illumination.
""")

# Image uploader for feature descriptors
uploaded_image = st.file_uploader("Upload an Image for Feature Descriptors", type=["jpg", "png", "jpeg"])
image = None  # Initialize the image variable

# Process the image if uploaded
if uploaded_image:
    image = np.array(Image.open(uploaded_image))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Image Uploaded Successfully!")

# Proceed only if an image is uploaded
if image is not None:
    # SIFT Feature Detection
    st.header("SIFT (Scale-Invariant Feature Transform)")
    st.write("""
    **SIFT** detects and describes local features in images.
    """)
    if st.button("Detect SIFT Features"):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            sift_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            st.image(sift_image, caption='SIFT Keypoints', use_column_width=True)
            st.write(f"Detected {len(keypoints)} keypoints using SIFT.")
        except cv2.error as e:
            st.error("SIFT is not available in this OpenCV build. Please use OpenCV-contrib-python.")

    # SURF Feature Detection
    st.header("SURF (Speeded-Up Robust Features)")
    st.write("""
    **SURF** is a faster version of SIFT for detecting and describing features.
    """)
    if st.button("Detect SURF Features"):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            surf = cv2.xfeatures2d.SURF_create(400)
            keypoints, descriptors = surf.detectAndCompute(gray, None)
            surf_image = cv2.drawKeypoints(image, keypoints, None, (0, 255, 0), 4)
            st.image(surf_image, caption='SURF Keypoints', use_column_width=True)
            st.write(f"Detected {len(keypoints)} keypoints using SURF.")
        except AttributeError:
            st.error("SURF is not available in this OpenCV build. Please ensure you have OpenCV-contrib-python installed.")

    # HOG Feature Descriptor
    st.header("HOG (Histogram of Oriented Gradients)")
    st.write("""
    **HOG** captures the appearance and shape of objects using gradient orientation histograms.
    """)
    if st.button("Apply HOG"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        h = hog.compute(gray)
        st.write(f'HOG Descriptor Shape: {h.shape}')
        st.image(image, caption='Original Image for HOG Descriptor', use_column_width=True)

# SIFT Mathematics
st.subheader("Mathematics Behind SIFT (Scale-Invariant Feature Transform)")
st.markdown("**SIFT** detects and describes local features in images that are invariant to scale and rotation:")
st.latex(r'''
    \text{DOG}(x, y, \sigma) = G(x, y, k\sigma) - G(x, y, \sigma)
''')
st.markdown("""
1. **Keypoint Localization**:
   Finds precise locations of keypoints by fitting a quadratic function to the difference of Gaussian images.

2. **Orientation Assignment**:
""")
st.latex(r'''
    \theta = \arctan\left(\frac{\partial I}{\partial y} \Big/ \frac{\partial I}{\partial x}\right)
''')
st.markdown("""
3. **Descriptor Generation**:
   - Generates descriptors by computing histograms of gradient orientations in regions around each keypoint.
""")

# SURF Mathematics
st.subheader("Mathematics Behind SURF (Speeded-Up Robust Features)")
st.markdown("**SURF** is a faster version of SIFT that uses integral images and approximates the Hessian matrix:")
st.latex(r'''
    \text{det}(H) = \left(\frac{\partial^2 I}{\partial x^2}\right) \left(\frac{\partial^2 I}{\partial y^2}\right) - \left(\frac{\partial^2 I}{\partial x \partial y}\right)^2
''')
st.markdown("""
- Describes regions around each keypoint using Haar wavelet responses, creating a 64-dimensional or 128-dimensional descriptor.
""")

# HOG Mathematics
st.subheader("Mathematics Behind HOG (Histogram of Oriented Gradients)")
st.markdown("**HOG** captures object shape and appearance using gradients:")
st.latex(r'''
    G_x = \frac{\partial I}{\partial x}, \quad G_y = \frac{\partial I}{\partial y}
''')
st.markdown("""
1. **Orientation Binning**:
""")
st.latex(r'''
    \theta = \arctan\left(\frac{G_y}{G_x}\right)
''')
st.markdown("""
2. **Block Normalization**:
   Groups cells into larger blocks and normalizes histograms to ensure invariance to illumination changes.

3. **Descriptor Vector**:
   Concatenates histograms to form a feature vector used for object detection.
""")
