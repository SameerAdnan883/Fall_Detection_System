# 🚨 Fall Detection System

An AI-powered real-time human fall detection system using **MediaPipe Pose** and **Random Forest Machine Learning**.

## 🎯 Project Objective

The system detects whether a person is experiencing a fall using body-pose landmarks extracted from a camera feed.

It is designed to provide real-time detection and an alarm when a fall is confirmed.

---

## 🧠 Technology Stack

- Python
- OpenCV
- MediaPipe Pose
- NumPy
- Pandas
- Scikit-learn
- Random Forest
- Streamlit
- Joblib

---

## 🔄 Detection Pipeline

```text
Camera
   ↓
MediaPipe Pose Detection
   ↓
Body Landmarks
   ↓
Feature Engineering
   ↓
Imputation + Scaling
   ↓
Random Forest
   ↓
Fall Probability
   ↓
Visibility Check
   ↓
Consecutive Frame Confirmation
   ↓
🚨 Fall Alarm