# 🌿 Plant Disease Detection using CNN

A deep learning web app that detects **38 plant diseases** from leaf images using a custom Convolutional Neural Network, built with TensorFlow/Keras and deployed via Streamlit.

---

## 📸 Demo

Upload a leaf photo → get instant disease prediction with confidence scores.

> Supported plants: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## 🧠 Model Architecture

Custom CNN built with Keras Sequential API:

```
Input (256×256×3)
│
├── Conv2D(16) → BatchNorm → MaxPool
├── Conv2D(32) → BatchNorm → MaxPool
├── Conv2D(64) → BatchNorm → MaxPool
├── Conv2D(64) → BatchNorm → MaxPool
├── Conv2D(64) → BatchNorm → MaxPool
│
├── GlobalAveragePooling2D
├── Dense(128, relu)
└── Dense(38, softmax)
```

**Training details:**
- Dataset: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (70,295 training images)
- Image size: 256×256
- Optimizer: Adam (lr=0.0005)
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

---

## 📁 Project Structure

```
├── main.py          # Streamlit web app
├── model.keras      # Trained model weights (download separately)
├── label.txt        # Class label reference
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install streamlit tensorflow pillow numpy
```

### 2. Download the model

Download `model.keras` and place it in the project root directory.

### 3. Run the app

```bash
streamlit run main.py
```

Then open `http://localhost:8501` in your browser.

---


## 🔧 How It Works

1. User uploads a leaf image (JPG/PNG/WebP)
2. Image is resized to **256×256** and normalized to **[0, 1]**
3. The CNN predicts probabilities across 38 classes
4. The app displays the top prediction + confidence + top-5 bar chart

---


## 🛠️ Tech Stack

- **Model:** TensorFlow / Keras
- **App:** Streamlit
- **Image processing:** Pillow, NumPy
- **Training environment:** Google Colab

---

