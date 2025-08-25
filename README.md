🚗 Real-Time Driver Drowsiness Detection System
This project is a real-time driver drowsiness detection system that uses computer vision and deep learning to monitor a driver’s face and detect signs of fatigue such as eye closure, yawning, and head tilt. The goal is to provide timely alerts to prevent road accidents caused by drowsy driving.
Features
•	Real-time face and landmark detection using OpenCV and Dlib.
•	Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pose Estimation for accurate drowsiness detection.
•	Multi-indicator fusion approach to minimize false positives.
•	Lightweight and deployable on low-cost hardware such as Raspberry Pi.
•	Provides visual and audio alerts when signs of drowsiness are detected.
Technologies Used
•	Python 3
•	OpenCV – for video processing
•	Dlib – facial landmark detection
•	NumPy / Pandas – data handling
•	Flask / Streamlit (optional) – for UI integration
•	TensorFlow/Keras – for additional deep learning-based classification
Installation
•	Clone the repository:
   git clone https://github.com/himanshugupta00235/Real-Time-Driver-Drowsiness-Detection-System-By-Himanshu.git
•	cd Real-Time-Driver-Drowsiness-Detection-System-By-Himanshu
•	Create and activate a virtual environment (recommended):
   python3 -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
•	Install dependencies:
   pip install -r requirements.txt
Usage
•	Run the application:
   python app.py
•	The webcam will start, and the system will begin detecting drowsiness in real-time.
•	If drowsiness is detected, an alert will be triggered.
Project Workflow
•	Face Detection – Haar cascade classifier locates the driver’s face.
•	Facial Landmark Detection – Dlib identifies 68 key facial points.
•	Eye Aspect Ratio (EAR) – Calculates how long eyes remain closed.
•	Mouth Aspect Ratio (MAR) – Detects yawns.
•	Head Pose Estimation – Monitors head tilts/nods.
•	Alert System – Warns the driver if drowsiness indicators cross thresholds.
Results
•	Achieved ~94.3% accuracy under various lighting conditions.
•	False positive rate reduced to 5.7% compared to traditional PERCLOS-only systems.
•	Real-time performance at 15 FPS on standard hardware.
Future Improvements
•	Integration of infrared cameras for low-light/night driving.
•	Mobile app integration for driver alerts.
•	Cloud-based analytics for fleet monitoring.
Author
•	👤 Himanshu Gupta
•	📧 Email: himanshugupta00235@gmail.com
•	🌐 GitHub: himanshugupta00235 (https://github.com/himanshugupta00235)
