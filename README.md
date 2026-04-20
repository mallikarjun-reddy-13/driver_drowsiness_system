# 🚗 Driver Drowsiness Detection using Deep Learning

## 📌 Overview

This project detects driver drowsiness in real-time using computer vision and deep learning.
The system monitors the driver's eyes through a webcam and triggers an alert if drowsiness is detected.

## 🎯 Problem Statement

Driver fatigue is one of the major causes of road accidents.
This system helps in reducing accidents by detecting eye closure and alerting the driver instantly.

## 🧠 Approach

The system works in the following steps:

1. Capture real-time video using webcam
2. Detect face using Haar Cascade
3. Detect left and right eyes
4. Preprocess eye images
5. Predict eye state (Open/Closed) using CNN model
6. Trigger alert if eyes remain closed for a certain duration

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy

## 📁 Project Structure

```
Driver_Drowsiness_Detection_AI/
│
├── detect_drowsiness_refactored.py        # Main execution file
├── drowiness_new7.h5           # Trained CNN model
├── model_training.ipynb        # Model training (optional)
├── README.md
├── requirements.txt
│
├── data/
│   ├── alarm.mp3
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_lefteye_2splits.xml
│   └── haarcascade_righteye_2splits.xml
```

## ▶️ How to Run

1. Clone the repository

```
git clone <your-repo-link>
cd Driver_Drowsiness_Detection_AI
```

2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Run the project

```
python detect_drowsiness_refactored.py
```

## 🚀 Features

* Real-time face and eye detection
* Deep learning-based eye classification
* Alert system using sound
* Frame-based drowsiness detection
* Lightweight and easy to run

## ⚠️ Limitations

* Performance depends on lighting conditions
* May misdetect when face is not clearly visible
* Accuracy depends on trained dataset

## 🔮 Future Improvements

* Use advanced models (ResNet, MobileNet)
* Add head pose detection
* Improve accuracy with larger dataset
* Build web UI using Streamlit
* Deploy on embedded systems

## 📌 Conclusion

This project demonstrates how AI and computer vision can be used to build real-time safety systems.
It highlights the practical application of deep learning in preventing accidents.# driver_drowsiness_system
# driver_drowsiness_system
