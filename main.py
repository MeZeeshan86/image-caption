import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
from PIL import Image

# Function to generate caption
def generate_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34, img_size=224):
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    image_features = feature_extractor.predict(img_array, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text.replace("startseq", "").replace("endseq", "").strip()

# Streamlit UI
def main():
    st.set_page_config(page_title="Image Caption Generator", page_icon="📷", layout="centered")

    # Inject custom CSS
    st.markdown("""
        <style>
        body {
            background: linear-gradient(135deg, #f9f9f9, #e6f7ff);
            font-family: 'Arial', sans-serif;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            font-size: 3em;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #34495e;
            font-size: 1.2em;
            margin-bottom: 30px;
        }
        .image-container {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        .caption-box {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            font-size: 1.2em;
            color: #2c3e50;
            margin-top: 20px;
        }
        .caption-box b {
            color: #2980b9;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title & subtitle
    st.markdown("<div class='title'>📷 Image Caption Generator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload an image and let AI describe it beautifully</div>", unsafe_allow_html=True)

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        temp_path = "uploaded_image.jpg"
        image.save(temp_path)

        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("✨ Generating caption..."):
            model_path = "models/model.keras"
            tokenizer_path = "models/tokenizer.pkl"
            feature_extractor_path = "models/feature_extractor.keras"
            caption = generate_caption(temp_path, model_path, tokenizer_path, feature_extractor_path)

        st.markdown(f"<div class='caption-box'><b>📝 Generated Caption:</b><br>{caption}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
