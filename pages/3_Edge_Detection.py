import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Edge Detection")

# Introduction to Edge Detection
st.header("Introduction to Edge Detection")
st.write("""
**Edge detection** is a technique used to identify boundaries within images, where the image intensity changes abruptly. It helps to highlight the structural features of an object, like edges, corners, and lines, which are essential for understanding the shape and structure of objects within the image.
""")

# Image uploader for edge detection
uploaded_image = st.file_uploader("Upload an Image for Edge Detection", type=["jpg", "png", "jpeg"])
image = None  # Initialize the image variable

# Process the image if uploaded
if uploaded_image:
    image = np.array(Image.open(uploaded_image))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Image Uploaded Successfully!")

# Proceed only if an image is uploaded
if image is not None:
    # Canny Edge Detection
    st.header("Canny Edge Detection")
    st.write("""
    **Canny Edge Detection** is a multi-step process involving noise reduction, gradient calculation, non-maximum suppression, and edge tracing through hysteresis.
    """)
    low_threshold = st.slider("Canny Lower Threshold", 0, 255, 100)
    high_threshold = st.slider("Canny Upper Threshold", 0, 255, 200)
    edges_canny = cv2.Canny(image, low_threshold, high_threshold)
    st.image(edges_canny, caption='Canny Edge Detection', use_column_width=True)

    # Laplacian of Gaussian (LOG)
    st.header("Laplacian of Gaussian (LOG)")
    st.write("""
    **Laplacian of Gaussian (LOG)** combines the Laplacian operator with a Gaussian filter for edge detection.
    """)
    log = cv2.Laplacian(cv2.GaussianBlur(image, (3, 3), 0), cv2.CV_64F)
    log_image = np.uint8(np.absolute(log))
    st.image(log_image, caption='Laplacian of Gaussian Edge Detection', use_column_width=True)

    # Difference of Gaussian (DOG)
    st.header("Difference of Gaussian (DOG)")
    st.write("""
    **Difference of Gaussian (DOG)** uses two Gaussian-blurred versions of an image with different sigma values and subtracts them.
    """)
    blurred1 = cv2.GaussianBlur(image, (5, 5), 1)
    blurred2 = cv2.GaussianBlur(image, (5, 5), 2)
    dog_image = cv2.subtract(blurred1, blurred2)
    st.image(dog_image, caption='Difference of Gaussian (DOG) Edge Detection', use_column_width=True)

# Canny Edge Detection Mathematics
st.subheader("Mathematics Behind Canny Edge Detection")
st.markdown("**Canny Edge Detection** is a multi-step process for detecting edges in images. It involves the following steps:")
st.markdown("1. **Noise Reduction**: A Gaussian filter is applied to the image to smooth it and reduce noise:")
st.latex(r'''
    G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
''')
st.markdown("where \(\sigma\) is the standard deviation of the Gaussian, controlling the amount of smoothing.")

st.markdown("2. **Gradient Calculation**: The gradients in the x and y directions are calculated using partial derivatives:")
st.latex(r'''
    I_x = \frac{\partial I}{\partial x}, \quad I_y = \frac{\partial I}{\partial y}
''')
st.latex(r'''
    G = \sqrt{I_x^2 + I_y^2}
''')
st.markdown("The gradient magnitude \(G\) indicates the strength of edges, and the angle of the gradient is:")
st.latex(r'''
    \theta = \arctan\left(\frac{I_y}{I_x}\right)
''')

st.markdown("3. **Non-Maximum Suppression**: Keeps only the local maxima in the direction of the gradient.")

st.markdown("4. **Hysteresis Thresholding**: Uses two thresholds, high and low, to trace edges:")
st.latex(r'''
    \text{If } G > \text{High Threshold, keep as edge. If } G < \text{Low Threshold, discard. Otherwise, keep if connected to a strong edge.}
''')

# Laplacian of Gaussian (LOG) Mathematics
st.subheader("Mathematics Behind Laplacian of Gaussian (LOG)")
st.markdown("**Laplacian of Gaussian (LOG)** is an edge detection technique that combines the Laplacian operator with Gaussian smoothing:")
st.markdown("1. **Gaussian Smoothing**: Smooths the image to reduce noise:")
st.latex(r'''
    G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
''')
st.markdown("2. **Laplacian Operator**: Measures the second-order derivatives, detecting areas where the intensity changes rapidly:")
st.latex(r'''
    \nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}
''')
st.markdown("3. **Laplacian of Gaussian (LOG)**: Applies the Laplacian operator to the Gaussian-smoothed image:")
st.latex(r'''
    \text{LOG}(x, y) = \nabla^2 (G(x, y) \ast I(x, y))
''')
st.markdown("where \(\ast\) denotes the convolution operation.")

# Difference of Gaussian (DOG) Mathematics
st.subheader("Mathematics Behind Difference of Gaussian (DOG)")
st.markdown("**Difference of Gaussian (DOG)** approximates the Laplacian of Gaussian by using two Gaussian-blurred versions of the image:")
st.latex(r'''
    \text{DOG}(x, y) = G(x, y, \sigma_1) \ast I(x, y) - G(x, y, \sigma_2) \ast I(x, y)
''')
st.markdown("where \(\sigma_1\) and \(\sigma_2\) are different standard deviations of the Gaussian filters, and \(\sigma_1 < \sigma_2\).")
st.markdown("This method highlights edges and textures by emphasizing regions where the intensity changes rapidly between scales.")

