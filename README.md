# 📷 Image Caption Generator using CNN & LSTM

This repository contains a **complete AI image captioning pipeline**:
1. **Model Training** – Train a CNN+LSTM model on the Flickr8k dataset to generate captions.
2. **Web App Deployment** – An interactive HTML/CSS-styled app that lets users upload images and get captions instantly.

---

## ✨ Features
- **End-to-End Workflow** – From dataset preprocessing to deployment.
- **Deep Learning Model** – CNN for image feature extraction + LSTM for sequential text generation.
- **Modern Web UI** – Gradient backgrounds, styled caption boxes, and responsive layout.
- **Multiple Image Formats** – Supports `.jpg`, `.jpeg`, and `.png`.
- **Reusable Models** – Save and load trained models for inference.

---

## 📊 Dataset
**Flickr8k Dataset**
- ~8,000 images with 5 human-written captions each.
- Captions describe scenes, objects, and actions.
- Ideal for small-scale image captioning experiments.

---

## ⚙️ Model Architecture
### Encoder (CNN)
- Pre-trained InceptionV3 or similar CNN model.
- Removes classification layer to output feature vectors.

### Tokenizer
- Maps words to integers for training and inference.
- Saved via `pickle` for consistent token mapping.

### Decoder (LSTM)
- Takes image features and partial captions as input.
- Predicts captions word-by-word until `<endseq>` token.

---

## 🛠 Training Workflow (`flickr8k-image-captioning-using-cnns-lstms.ipynb`)
1. **Data Preparation** – Load images and captions, clean text, remove punctuation, lowercase words, tokenize, and pad sequences.  
2. **Feature Extraction** – Extract image features with a CNN and save them for faster training.  
3. **Model Definition** – Merge CNN features with embedded caption sequences and feed them into an LSTM decoder.  
4. **Training** – Use categorical cross-entropy loss and monitor validation accuracy and loss.  
5. **Save Models** – Store `model.keras`, `feature_extractor.keras`, and `tokenizer.pkl` for deployment.

---

## 🚀 App Usage (`main.py`)
```bash
# 1️⃣ Clone the Repository
git clone https://github.com/your-username/image-caption-generator.git
cd image-caption-generator

# 2️⃣ Install Dependencies
pip install -r requirements.txt

# 3️⃣ Add Trained Models
# Place trained model files into the models/ directory:
models/
 ├── model.keras
 ├── feature_extractor.keras
 └── tokenizer.pkl

# 4️⃣ Run the Application
streamlit run main.py

# 📂 Project Structure
├── main.py                      # Web app interface + caption generation
├── flickr8k-image-captioning... # Model training notebook
├── models/
│   ├── model.keras
│   ├── feature_extractor.keras
│   └── tokenizer.pkl
├── requirements.txt
└── README.md
