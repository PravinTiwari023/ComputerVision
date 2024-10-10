import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Line Detection")

# Introduction to Line Detection
st.header("Introduction to Line Detection")
st.write("""
**Line detection** is used to identify straight lines in an image. It is particularly useful in applications like road lane detection for autonomous vehicles and structural analysis.
""")

# Image uploader for line detection
uploaded_image = st.file_uploader("Upload an Image for Line Detection", type=["jpg", "png", "jpeg"])
image = None  # Initialize the image variable

# Process the image if uploaded
if uploaded_image:
    image = np.array(Image.open(uploaded_image))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Image Uploaded Successfully!")

# Proceed only if an image is uploaded
if image is not None:
    # Hough Transform for Line Detection
    st.header("Hough Transform for Line Detection")
    st.write("""
    **Hough Transform** is a technique for detecting straight lines by transforming points into a parameter space.
    """)
    canny_edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(canny_edges, 1, np.pi/180, 100)

    # Draw lines on the image
    hough_image = image.copy()
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        st.image(hough_image, caption='Hough Transform Line Detection', use_column_width=True)
    else:
        st.write("No lines detected. Try adjusting the Canny edge detection thresholds.")

# Hough Transform Mathematics
st.subheader("Mathematics Behind Hough Transform")
st.markdown("The **Hough Transform** is used to detect straight lines by converting points in Cartesian space \((x, y)\) into a parameter space \((\rho, \theta)\):")
st.latex(r'''
    \rho = x \cdot \cos(\theta) + y \cdot \sin(\theta)
''')
st.markdown("""
- **\(\rho\)**: Distance from the origin to the line.
- **\(\theta\)**: Angle of the line with respect to the x-axis.
""")
st.markdown("""
1. **Hough Space**:
   - A point in Cartesian space corresponds to a sinusoidal curve in Hough space.
   - Each point along a line in the image contributes to a sinusoid in Hough space, and the intersection of these sinusoids indicates the presence of a line.

2. **Finding Intersections**:
   - Peaks in the Hough accumulator space correspond to the parameters \((\rho, \theta)\) of detected lines.
""")

