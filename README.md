<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:141E30,100:243B55&height=200&section=header&text=Driver%20Drowsiness%20System&fontSize=35&fontColor=ffffff"/>

### 🚗 Real-Time Driver Monitoring using AI & Computer Vision  

<img src="https://readme-typing-svg.herokuapp.com?color=00F7FF&center=true&vCenter=true&lines=Detecting+Fatigue+in+Real-Time;Computer+Vision+Project;AI+Safety+System"/>

<img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"/>
<img src="https://img.shields.io/badge/OpenCV-Vision-green?style=for-the-badge&logo=opencv"/>
<img src="https://img.shields.io/badge/TensorFlow-AI-orange?style=for-the-badge&logo=tensorflow"/>

</div>


## 🧠 About the Project  

This is a **real-time driver drowsiness detection system** built using computer vision and deep learning.  

It continuously monitors the driver's eyes through a webcam and triggers alerts when signs of fatigue are detected.  

> ⚠️ Designed as a safety-focused system to reduce accident risk caused by drowsy driving.


## ⚡ Key Features  

```diff
+ Real-time face & eye tracking
+ CNN-based eye state classification
+ Instant drowsiness alert (sound + recording)
+ FPS + status monitoring
+ Lightweight and fast

```Works
Detects face using OpenCV
Extracts eye regions using cascade classifiers
Passes eye images into CNN model
Tracks eye state over frames
If eyes remain closed → triggers alert system.

```clone 

git clone https://github.com/mallikarjun-reddy-13/driver_drowsiness_system.git
cd driver_drowsiness_system

venv310\Scripts\activate.bat
pip install -r requirements_clean.txt

python detect_face_drowsiness_fixed.py

Project Structure
driver_drowsiness_system/
│── detect_face_drowsiness_fixed.py
│── drowiness_new7.h5
│── data/
│── requirements_clean.txt
│── steps.md

🧪 Testing the System
Start the program
Sit in front of webcam
Close your eyes for a few seconds
Expected Output:
🔴 “DROWSY” alert
🔊 Alarm sound
🎥 Recording starts

📊 Model Details
CNN trained on eye dataset
Input: 145×145 grayscale images
Output classes: 0–3 (3 = open)
Pre-trained .h5 model included

🔧 Troubleshooting
<details> <summary>Common Issues</summary>

Missing modules

pip install opencv-python tensorflow

VSCode errors

Select correct interpreter (venv310)
Reload window

Camera issue

cv2.VideoCapture(1)

No sound
Check data/alarm.mp3

</details>


🙌 Final Thoughts

Built through real debugging, testing, and iteration.
Not perfect — but functional and extensible.

If this helped you, consider giving a ⭐

<div align="center">

💡 Built for safety, learning, and real-world application

</div> <img src="https://capsule-render.vercel.app/api?type=waving&color=0:243B55,100:141E30&height=120&section=footer"/>

