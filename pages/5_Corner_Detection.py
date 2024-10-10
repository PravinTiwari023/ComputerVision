import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Corner Detection")

# Introduction to Corner Detection
st.header("Introduction to Corner Detection")
st.write("""
**Corner detection** is used to find points in an image where the intensity gradient changes significantly, like corners or intersections.
""")

# Image uploader for corner detection
uploaded_image = st.file_uploader("Upload an Image for Corner Detection", type=["jpg", "png", "jpeg"])
image = None  # Initialize the image variable

# Process the image if uploaded
if uploaded_image:
    image = np.array(Image.open(uploaded_image))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Image Uploaded Successfully!")

# Proceed only if an image is uploaded
if image is not None:
    # Harris Corner Detection
    st.header("Harris Corner Detection")
    st.write("""
    **Harris Corner Detection** finds corners in an image by evaluating intensity gradients.
    """)
    if st.button("Apply Harris Corner Detection"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        harris_corners = cv2.dilate(harris_corners, None)
        harris_image = image.copy()
        harris_image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
        st.image(harris_image, caption='Harris Corner Detection', use_column_width=True)

    # Hessian Affine Approximation
    st.header("Hessian Affine Approximation")
    st.write("""
    **Hessian Affine** detects stable regions in an image by analyzing curvature using the Hessian matrix.
    """)
    if st.button("Apply Hessian Affine Approximation"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        hessian_affine_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
        st.image(hessian_affine_image, caption='Hessian Affine Approximation', use_column_width=True)
        st.write(f"Detected {len(keypoints)} keypoints using Hessian-based detection.")

# Harris Corner Detection Mathematics
st.subheader("Mathematics Behind Harris Corner Detection")
st.markdown("The **Harris Corner Detection** algorithm identifies points in an image where there is a significant change in gradient in multiple directions:")
st.latex(r'''
    I_x = \frac{\partial I}{\partial x}, \quad I_y = \frac{\partial I}{\partial y}
''')
st.markdown("""
1. **Structure Tensor**:
""")
st.latex(r'''
    M = 
    \begin{pmatrix}
    I_x^2 & I_x I_y \\
    I_x I_y & I_y^2
    \end{pmatrix}
''')
st.markdown("""
2. **Corner Response Function**:
""")
st.latex(r'''
    R = \text{det}(M) - k \cdot (\text{trace}(M))^2
''')
st.markdown("""
- \(\text{det}(M)\): Determinant of \(M\).
- \(\text{trace}(M)\): Sum of the diagonal elements of \(M\).
- \(k\): Sensitivity parameter (typically between 0.04 and 0.06).
""")

# Hessian Affine Mathematics
st.subheader("Mathematics Behind Hessian Affine Approximation")
st.markdown("**Hessian Affine** identifies stable regions in an image by analyzing the curvature using the **Hessian matrix**:")
st.latex(r'''
    H =
    \begin{pmatrix}
    \frac{\partial^2 I}{\partial x^2} & \frac{\partial^2 I}{\partial x \partial y} \\
    \frac{\partial^2 I}{\partial x \partial y} & \frac{\partial^2 I}{\partial y^2}
    \end{pmatrix}
''')
st.markdown("""
- The determinant and trace of \(H\) help detect blobs and corners:
""")
st.latex(r'''
    \text{det}(H) = \lambda_1 \lambda_2, \quad \text{trace}(H) = \lambda_1 + \lambda_2
''')
st.markdown("""
  - \(\lambda_1\) and \(\lambda_2\) are the eigenvalues of \(H\).
  - Points with high curvature in all directions correspond to stable features.
""")
