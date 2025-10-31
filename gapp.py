import streamlit as st
from PIL import Image
import tempfile
import numpy as np
import cv2
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Page config

st.set_page_config(page_title="Garbage Detection & Classification", page_icon="🗑️", layout="wide")
st.title("🗑️ Garbage Detection & Classification Dashboard")
st.write("Upload an image to detect garbage objects and classify them simultaneously.")


# Load Models

@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")  # YOLO detection model

@st.cache_resource
def load_cnn_model():
    return load_model("garbage_classifier_final.keras")  # CNN classification model

yolo_model = load_yolo_model()
cnn_model = load_cnn_model()
class_labels = ["Plastic", "Paper", "Metal", "Glass", "Organic", "Other"]  # Replace with actual


# File uploader

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    
    # YOLO Detection
   
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        temp_path = temp.name

    yolo_results = yolo_model(temp_path)

    # Convert to OpenCV image for drawing
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.subheader("🔍 Garbage Detection")

    if len(yolo_results[0].boxes) == 0:
        st.warning("No objects detected by YOLO!")
    else:
        for i, box in enumerate(yolo_results[0].boxes.xyxy):  # xyxy = [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(img_cv, f"Object {i+1}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    
    # CNN Classification
    
    img_resized = image.resize((384, 512))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = cnn_model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Overlay CNN label on the image
    cv2.putText(img_cv, f"Class: {predicted_class} ({confidence:.0f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)  # Red label

    final_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    
    # Side-by-side display
    
    st.subheader("🔍 Result Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(final_img, caption="Detected & Classified Garbage", use_column_width=True)

    st.write(f"**Prediction (CNN Classification):** {predicted_class} ({confidence:.2f}%)")


    # Download Button
    
    output_path = "garbage_detected_classified.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    with open(output_path, "rb") as file:
        st.download_button(
            label="⬇️ Download Result Image",
            data=file,
            file_name="garbage_detected_classified.jpg",
            mime="image/jpeg"
        )
