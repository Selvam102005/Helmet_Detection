🏍️ AI-Powered Helmet Detection & Traffic Safety System
An automated computer vision solution designed to enhance road safety by detecting motorcycle riders and verifying helmet compliance in real-time. This system uses a sophisticated Two-Stage Deep Learning Pipeline and is deployed as a user-friendly web application.
🌟 Key Features
Automated Detection: Real-time localization of motorcycles and riders using YOLOv8.
High-Precision Classification: Specialized ResNet50 head-classifier to distinguish between "Helmet" and "No Helmet."
Big Data Trained: Trained on a massive merged dataset of 24,000+ traffic images for robust performance.
Interactive Web Portal: Built with Streamlit for easy image uploading and instant visual reporting.
Performance Metrics: Detailed dashboard showing "Safe Riders" vs. "Violations."
🧠 System Architecture
The system operates using a modular ensemble approach:
Stage 1 - The Detector (YOLOv8s): Scans the input for motorcycles and riders. It is optimized to work in crowded urban traffic environments.
Stage 2 - The Classifier (ResNet50): The system crops the head region of every detected rider and passes it to this specialist model, which determines safety compliance.
The Frontend (Streamlit): An intuitive interface that handles image processing and displays annotated results with a high-accuracy confidence score.
🚀 Installation & Setup
1. Requirements
Ensure you have Python 3.10 or 3.11 installed. Install the necessary libraries using pip:
code
Bash
pip install ultralytics streamlit torch torchvision opencv-python pandas pillow
2. Model Weights
Place your trained "AI Brains" in the following directory:
models/weights/yolo_best.pt (The Detector)
models/weights/helmet_classifier_best.pt (The Classifier)
3. Running the Application
To launch the web-based monitoring portal, run:
code
Bash
streamlit run app.py
Then open http://localhost:8501 in your web browser.
📂 Project Structure
app.py: The main web application script.
src/: Core logic folder.
simple_pipeline.py: Orchestrates the flow between YOLO and the CNN.
detector.py: Handles YOLOv8 inference.
classifier.py: Handles ResNet classification logic.
configs/:
train_config.py: Centralized thresholds (Current Sensitivity: 0.45).
models/weights/: Storage for pre-trained PyTorch models.
outputs/: Directory for saved analysis results.
📊 Dataset & Training
Total Training Images: 24,703
Total Classifier Samples: 16,212 (Balanced Helmet/No-Helmet)
Detection Accuracy: 88.6% mAP@0.5
Training Hardware: NVIDIA Tesla T4 (Google Colab)
Inference Hardware: Optimized for 4GB VRAM local GPUs or standard CPUs.
🛠️ Configuration
You can adjust the sensitivity of the AI by editing configs/train_config.py:
YOLO_CONF_THRESH: Set to 0.45 to ignore distant background objects.
HELMET_CONF_THRESHOLD: Set to 0.65 for strict violation detection.
📜 Acknowledgments
Ultralytics for the YOLOv8 framework.
Roboflow for providing access to diverse traffic datasets.
Streamlit for the web deployment framework.
Author: Selvam S
Project Status: Completed / Professional Prototype
Date: March 2026