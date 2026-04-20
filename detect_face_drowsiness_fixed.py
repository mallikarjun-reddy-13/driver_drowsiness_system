#!/usr/bin/env python3
"""
Driver Drowsiness Detection System - Pylance Fixed Version.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import tensorflow as tf
import pygame
import time
import datetime
import sys

class DrowsinessDetector:
    EYE_IMAGE_SIZE = 145
    DROWSY_FRAMES_THRESHOLD = 10
    TARGET_FPS = 30.0

    def __init__(self, model_path: str = "drowiness_new7.h5", alarm_sound_path: str = "data/alarm.mp3"):
        self.face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
        self.left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
        self.right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
        
        self.model = tf.keras.models.load_model(model_path)
        print(f"Loaded model: {model_path}")
        
        pygame.mixer.pre_init(22050, -16, 2, 512)
        pygame.mixer.init()
        self.alarm_sound_path = Path(alarm_sound_path)
        
        self.drowsy_frames = 0
        self.alarm_active = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.prev_fps_time = 0.0
        self.fps = 0.0
        self.left_eye_open = True
        self.right_eye_open = True
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            sys.exit(1)

    def classify_eye_state(self, eye_roi: np.ndarray) -> bool:
        eye_resized = cv2.resize(eye_roi, (self.EYE_IMAGE_SIZE, self.EYE_IMAGE_SIZE))
        eye_normalized = eye_resized.astype('float32') / 255.0
        eye_expanded = np.expand_dims(tf.keras.utils.img_to_array(eye_normalized), axis=0)
        
        prediction = self.model.predict(eye_expanded, verbose=0)[0]
        is_open = np.argmax(prediction) == 3
        return is_open

    def detect_eyes_in_face(self, gray_roi: np.ndarray, color_roi: np.ndarray) -> Tuple[bool, bool]:
        left_eye_open = True
        right_eye_open = True
        
        left_eyes = self.left_eye_cascade.detectMultiScale(gray_roi, 1.3, 5)
        for ex, ey, ew, eh in left_eyes[:1]:
            left_eye_roi = color_roi[ey:ey+eh, ex:ex+ew]
            left_eye_open = self.classify_eye_state(left_eye_roi)
            color = (0, 255, 0) if left_eye_open else (0, 0, 255)
            cv2.rectangle(color_roi, (ex, ey), (ex+ew, ey+eh), color, 2)
        
        right_eyes = self.right_eye_cascade.detectMultiScale(gray_roi, 1.3, 5)
        for ex, ey, ew, eh in right_eyes[:1]:
            right_eye_roi = color_roi[ey:ey+eh, ex:ex+ew]
            right_eye_open = self.classify_eye_state(right_eye_roi)
            color = (0, 255, 0) if right_eye_open else (0, 0, 255)
            cv2.rectangle(color_roi, (ex, ey), (ex+ew, ey+eh), color, 2)
        
        return left_eye_open, right_eye_open

    def handle_drowsiness_alert(self, frame: np.ndarray, both_eyes_closed: bool) -> None:
        if both_eyes_closed:
            self.drowsy_frames += 1
            if self.drowsy_frames >= self.DROWSY_FRAMES_THRESHOLD and not self.alarm_active:
                cv2.putText(frame, "DROWSINESS ALERT!", (100, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                self.trigger_alert(frame)
        else:
            self.drowsy_frames = 0
            self.stop_alert()

    def trigger_alert(self, frame: np.ndarray):
        try:
            pygame.mixer.music.load(self.alarm_sound_path)
            pygame.mixer.music.play(-1)
        except Exception as e:
            print(f"Audio error: {e}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{timestamp}.mp4"
        self.video_writer = cv2.VideoWriter(filename, fourcc, self.TARGET_FPS, (int(frame.shape[1]), int(frame.shape[0])))
        print(f"Recording to {filename}")
        self.alarm_active = True

    def stop_alert(self):
        pygame.mixer.music.stop()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print("Alert video saved")
        self.alarm_active = False

    def draw_ui_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        both_open = self.left_eye_open and self.right_eye_open
        status = "DROWSY" if not both_open else "AWAKE"
        color = (0, 0, 255) if status == "DROWSY" else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Drowsy Frames: {self.drowsy_frames}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        curr_time = time.time()
        if curr_time - self.prev_fps_time > 0.5:
            self.fps = 1.0 / (curr_time - self.prev_fps_time)
            self.prev_fps_time = curr_time
        cv2.putText(frame, f"FPS: {int(self.fps)}", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, "Driver Drowsiness Detection", (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        if self.video_writer is not None:
            cv2.putText(frame, "RECORDING", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

    def run(self):
        print("Press 'q' to quit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            self.left_eye_open = True
            self.right_eye_open = True
            
            for fx, fy, fw, fh in faces:
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
                face_gray = gray[fy:fy+fh, fx:fx+fw]
                face_color = frame[fy:fy+fh, fx:fx+fw]
                self.left_eye_open, self.right_eye_open = self.detect_eyes_in_face(face_gray, face_color)
            
            both_open = self.left_eye_open and self.right_eye_open
            self.handle_drowsiness_alert(frame, not both_open)
            
            if self.video_writer is not None:
                self.video_writer.write(frame)
            
            frame = self.draw_ui_overlay(frame)
            cv2.imshow("Drowsiness Detector", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        self.stop_alert()
        pygame.mixer.quit()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()

