# 🐶🐱 Dog vs. Cat Image Classification using SVM

A machine learning pipeline that implements a Support Vector Machine (SVM) configuration to ingest, preprocess, resize, and classify structural image arrays of cats and dogs.

## 🚀 Pipeline Features
* **Automated Image Ingestion:** Pulls the official Cats and Dogs dataset dynamically via `kagglehub`.
* **Computer Vision Preprocessing:** Leverages `OpenCV` to load images, convert them to grayscale, and resize them uniformly to standard pixel matrices.
* **Feature Flattening:** Normalizes and flattens 2D pixel structures into 1D feature arrays optimized for hyper-dimensional boundary tracking.
* **Support Vector Machine Classification:** Implements an optimized `SVC` model with regularized margin boundaries to separate animal classes.
* **Performance Insights:** Generates clean confusion matrices, precision/recall metrics, and test sample verification showcases.

## 📊 Performance Showcase
The optimization metrics, classification margins, and sample model outputs are automatically computed and rendered into your local storage outputs folder.

### 🪐 Evaluation & Matrix Breakdown
![SVM Model Performance](outputs/task3_svm_dashboard.jpg)

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Core Packages:** Scikit-Learn, OpenCV (opencv-python), NumPy, Pandas, Matplotlib, Seaborn, Kagglehub
