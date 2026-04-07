# 🏍️ AI-Powered Helmet Detection & Traffic Safety System

An automated computer vision solution designed to enhance road safety by detecting motorcycle riders and verifying helmet compliance in real-time.

This system uses a **Two-Stage Deep Learning Pipeline** and is deployed as a **user-friendly web application**.

---

## 🌟 Key Features

* 🚀 **Automated Detection**
  Real-time localization of motorcycles and riders using **YOLOv8**

* 🎯 **High-Precision Classification**
  ResNet50-based classifier to detect **Helmet / No Helmet**

* 📊 **Big Data Trained**
  Trained on **24,000+ traffic images** for strong generalization

* 💻 **Interactive Web Portal**
  Built with **Streamlit** for easy image upload and results

* 📈 **Performance Dashboard**
  Displays **Safe Riders vs Violations**

---

## 🧠 System Architecture

### 🔹 Stage 1 — Detector (YOLOv8s)

* Detects motorcycles and riders
* Works efficiently in dense traffic conditions

### 🔹 Stage 2 — Classifier (ResNet50)

* Crops rider head region
* Classifies into:

  * ✅ Helmet
  * ❌ No Helmet

### 🔹 Frontend — Streamlit

* Upload images
* Displays annotated results
* Shows confidence scores

---

## 🚀 Installation & Setup

### 📌 Requirements

* Python **3.10 or 3.11**

### 📦 Install Dependencies

```bash
pip install ultralytics streamlit torch torchvision opencv-python pandas pillow
```

---

## 🧠 Model Weights

Place your trained models inside:

```
models/weights/
├── yolo_best.pt
└── helmet_classifier_best.pt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

## 📂 Project Structure

```
HelmetProject/
│── app.py
│── src/
│   ├── simple_pipeline.py
│   ├── detector.py
│   ├── classifier.py
│
│── configs/
│   └── train_config.py
│
│── models/
│   └── weights/
│
│── outputs/
│
│── README.md
│── requirements.txt
```

---

## 📊 Dataset & Training

* 📸 Total Images: **24,703**
* 🧠 Classifier Samples: **16,212**
* 🎯 Detection Accuracy: **88.6% mAP@0.5**

### ⚙️ Hardware

* Training: **NVIDIA Tesla T4 (Colab)**
* Inference: **4GB VRAM GPU / CPU**

---

## 🛠️ Configuration

Edit:

```
configs/train_config.py
```

| Parameter             | Value | Description                |
| --------------------- | ----- | -------------------------- |
| YOLO_CONF_THRESH      | 0.45  | Ignore distant objects     |
| HELMET_CONF_THRESHOLD | 0.65  | Strict violation detection |

---

## 🙌 Acknowledgments

* Ultralytics (YOLOv8)
* Roboflow (Datasets)
* Streamlit (Frontend)

---

## 📌 Project Status

✅ Completed — Professional Prototype
📅 March 2026
