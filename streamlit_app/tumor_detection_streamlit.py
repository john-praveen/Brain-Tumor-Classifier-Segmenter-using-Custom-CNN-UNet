import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Brain Tumor Detection & Segmentation", layout="wide")

# Custom loss functions
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

custom_objects = {
    'dice_coefficient': dice_coefficient,
    'dice_loss': dice_loss,
    'combined_loss': combined_loss
}

@st.cache_resource
def load_all_models():
    return {
        "classifier": load_model("tumor_type_classifier_cnn.h5", custom_objects=custom_objects),
        "glioma": load_model("glioma_segmentation.h5", custom_objects=custom_objects),
        "meningioma": load_model("Meningioma_segmentation.h5", custom_objects=custom_objects),
        "pituitary": load_model("pitutary_segmentation.h5", custom_objects=custom_objects)
    }

def preprocess(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(img, (128, 128))
    img_norm = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_norm, axis=0), img_resized

def predict_type(model, img):
    preds = model.predict(img)[0]
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    return labels[np.argmax(preds)], np.max(preds), preds

def segment(model, img):
    pred = model.predict(img)[0, :, :, 0]
    mask = (pred > 0.5).astype(np.uint8)
    return pred, mask

def overlay(image, mask):
    overlay_img = image.copy()
    red_mask = np.zeros_like(overlay_img)
    red_mask[:, :, 0] = mask * 255
    return cv2.addWeighted(overlay_img, 0.6, red_mask, 0.4, 0)

def main():
    st.title("🧠 Brain Tumor Detection & Segmentation")
    st.write("Upload an MRI image. The app will detect tumor type and perform segmentation if applicable.")

    models = load_all_models()

    file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png", "bmp"])

    if file:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_input, img_display = preprocess(image)
        tumor_type, confidence, probs = predict_type(models["classifier"], img_input)

        st.subheader("Prediction Results")
        if tumor_type == "notumor":
            st.success(f"No Tumor Detected with {confidence:.2%} confidence.")
        else:
            st.error(f"{tumor_type.capitalize()} Tumor Detected with {confidence:.2%} confidence.")
            st.subheader("Segmentation")
            seg_model = models[tumor_type]
            prob_mask, bin_mask = segment(seg_model, img_input)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img_display, caption="Original Image", use_column_width=True)
            with col2:
                st.image(prob_mask, caption="Probability Mask", use_column_width=True, clamp=True)
            with col3:
                st.image(overlay(img_display, bin_mask), caption="Overlay", use_column_width=True)

            tumor_percent = 100.0 * np.sum(bin_mask) / bin_mask.size
            st.info(f"Tumor covers approximately {tumor_percent:.2f}% of the image.")

if __name__ == "__main__":
    main()