import streamlit as st 
import cv2
import numpy as np
from PIL import Image
import pickle

st.title("🖼️ Image Processing & Classification ")

# --- Load pre-trained model and accuracy ---
with open("image_model.pkl", "rb") as f:
    model, class_names = pickle.load(f)

with open("accuracy.pkl", "rb") as f:
    acc = pickle.load(f)

st.header("🔬 Model Performance")
st.metric("📊 Accuracy", f"{acc*100:.2f}%")


# --- Keep image persistent using session_state ---
if "img_cv" not in st.session_state:
    st.session_state.img_cv = None
if "image" not in st.session_state:
    st.session_state.image = None

# Step 1: User uploads image directly
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load with PIL
    st.session_state.image = Image.open(uploaded_file)

    # Reset pointer before reading for OpenCV
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    st.session_state.img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.success("✅ Image uploaded successfully!")

# Utility function for saving images
def save_image_button(img, filename="processed.png", is_gray=False):
    is_success, buffer = cv2.imencode(".png", img)
    if is_success:
        st.download_button(
            label="💾 Save Image",
            data=buffer.tobytes(),
            file_name=filename,
            mime="image/png"
        )

# Only continue if image is loaded
if st.session_state.img_cv is not None:
    st.image(st.session_state.image, caption="Original Image", use_container_width=True)

    # Step 2: User selects operation
    operation = st.selectbox(
        "Choose an operation:",
        ["Sharpen", "Crop", "Blur", "Rotate", "Zoom", "Change Colour"]
    )

    img_cv = st.session_state.img_cv.copy()  # work on a copy

    if operation == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(img_cv, -1, kernel)
        st.image(sharpened, channels="BGR", caption="🔎 Sharpened Image")
        save_image_button(sharpened, "sharpened.png")

    elif operation == "Crop":
        h, w, _ = img_cv.shape
        st.info("👉 Use sliders to crop the image")

        x1 = st.slider("X1 (Left)", 0, w-1, 0)
        x2 = st.slider("X2 (Right)", 0, w, w)
        y1 = st.slider("Y1 (Top)", 0, h-1, 0)
        y2 = st.slider("Y2 (Bottom)", 0, h, h)

        if x1 < x2 and y1 < y2:
            cropped = img_cv[y1:y2, x1:x2]
            st.image(cropped, channels="BGR", caption=f"✂️ Cropped Image ({x2-x1}x{y2-y1})")
            save_image_button(cropped, "cropped.png")
        else:
            st.warning("⚠️ Adjust sliders so X1 < X2 and Y1 < Y2")

    elif operation == "Blur":
        k = st.slider("Blur intensity (odd number)", 1, 25, 5, step=2)
        blurred = cv2.GaussianBlur(img_cv, (k, k), 0)
        st.image(blurred, channels="BGR", caption="🌫️ Blurred Image")
        save_image_button(blurred, "blurred.png")

    elif operation == "Rotate":
        angle = st.slider("Rotation Angle", -180, 180, 0)
        h, w = img_cv.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        rotated = cv2.warpAffine(img_cv, M, (w, h))
        st.image(rotated, channels="BGR", caption=f"🔄 Rotated {angle}°")
        save_image_button(rotated, "rotated.png")

    elif operation == "Zoom":
        scale = st.slider("Zoom scale (%)", 100, 300, 100) / 100
        h, w = img_cv.shape[:2]
        new_w, new_h = int(w / scale), int(h / scale)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        zoomed = img_cv[start_y:start_y+new_h, start_x:start_x+new_w]
        zoomed = cv2.resize(zoomed, (w, h))  # resize back to original size
        st.image(zoomed, channels="BGR", caption=f"🔍 Zoomed {scale:.2f}x")
        save_image_button(zoomed, "zoomed.png")

    elif operation == "Change Colour":
        color_option = st.radio("Choose colour mode:", ["Grey", "Black & White"])

        if color_option == "Grey":
            grey = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            st.image(grey, caption="🖤 Grey Image", channels="GRAY")
            save_image_button(grey, "grey.png", is_gray=True)

        elif color_option == "Black & White":
            grey = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(grey, 128, 255, cv2.THRESH_BINARY)
            st.image(bw, caption="⬛⬜ Black & White Image", channels="GRAY")
            save_image_button(bw, "black_white.png", is_gray=True)

#1  env\Scripts\activate 


#2 pip install scikit-learn


#3 streamlit run script.py
#C:/Users/Nancy/Desktop/cv/data/Cat/403.jpg
