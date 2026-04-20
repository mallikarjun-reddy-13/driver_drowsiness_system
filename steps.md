# 🚗 Driver Drowsiness Detection using AI

## 🛠️ What I Did (Step-by-Step)

### 🔹 Step 1: Project Setup

* Downloaded project source code
* Opened in VS Code
* Checked folder structure and files

### 🔹 Step 2: Created Virtual Environment

```bash
python -m venv venv310
venv310\Scripts\activate
```

### 🔹 Step 3: Installed Required Libraries

```bash
pip install tensorflow opencv-python numpy playsound==1.2.2
```

### 🔹 Step 4: Fixed Import Errors

* Replaced:

```python
from keras.models import load_model
```

➡️ with:

```python
from tensorflow.keras.models import load_model
```

* Updated all `keras` imports to `tensorflow.keras`

---

### 🔹 Step 5: Fixed Python Compatibility Issue

* Faced issue with Python 3.12
* Installed Python 3.10
* Created new environment (`venv310`)
* Reinstalled dependencies

---

### 🔹 Step 6: Fixed Missing Modules

* Installed `playsound`
* Ensured all required packages are available

---

### 🔹 Step 7: Fixed Camera Issues

* Camera not opening initially
* Tried different indexes (0, 1, 2)
* Added auto-detection logic for camera

---

### 🔹 Step 8: Fixed Model Logic

* Corrected prediction logic:

```python
status = np.argmax(pred, axis=1)[0]
```

* Fixed condition:

```python
if status1 == 0 and status2 == 0:
```

---

### 🔹 Step 9: Improved Detection Logic

* Ignored false detection when eyes are not visible
* Adjusted frame threshold for faster alert

---

### 🔹 Step 10: UI Improvements

* Added status text (AWAKE / DROWSY)
* Added frame counter
* Added project title on screen
* Improved visual overlay

---

### 🔹 Step 11: Cleaned Project Structure

* Removed unnecessary files and folders
* Kept only required files:

  * Main script
  * Model file
  * Haarcascade files
  * Alarm sound

---

### 🔹 Step 12: Tested the System

* Verified:

  * Webcam works
  * Face detection works
  * Eye detection works
  * Alert triggers correctly

---

## 🧠 How It Works

1. Webcam captures video
2. Face detected using Haar Cascade
3. Eyes extracted from face
4. Eye images resized and normalized
5. CNN model predicts:

   * 0 → Closed
   * 1 → Open
6. If both eyes closed for multiple frames → Alert triggered

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* OpenCV
* NumPy
