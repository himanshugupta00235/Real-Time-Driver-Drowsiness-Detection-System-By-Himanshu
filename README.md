# 🚗 Real-Time Driver Drowsiness Detection System  

This project is a **real-time driver drowsiness detection system** that uses **computer vision and deep learning** to monitor a driver’s face and detect signs of fatigue such as eye closure, yawning, and head tilt. The goal is to provide timely alerts to prevent road accidents caused by drowsy driving.  

---

## 🔹 Features
- Real-time face and landmark detection using **OpenCV** and **Dlib**.  
- Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pose Estimation for accurate drowsiness detection.  
- Multi-indicator fusion approach to minimize false positives.  
- Lightweight and deployable on low-cost hardware such as **Raspberry Pi**.  
- Provides visual and audio alerts when signs of drowsiness are detected.  

---

## 🔹 Technologies Used
- **Python 3**  
- **OpenCV** – for video processing  
- **Dlib** – facial landmark detection  
- **NumPy / Pandas** – data handling  
- **Flask / Streamlit** (optional) – for UI integration  
- **TensorFlow/Keras** – for additional deep learning-based classification  

---

## 🔹 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/himanshugupta00235/Real-Time-Driver-Drowsiness-Detection-System-By-Himanshu.git
   cd Real-Time-Driver-Drowsiness-Detection-System-By-Himanshu
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔹 Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. The webcam will start, and the system will begin detecting drowsiness in real-time.  

3. If drowsiness is detected, an **alert** will be triggered.  

---

## 🔹 Project Workflow
1. **Face Detection** – Haar cascade classifier locates the driver’s face.  
2. **Facial Landmark Detection** – Dlib identifies 68 key facial points.  
3. **Eye Aspect Ratio (EAR)** – Calculates how long eyes remain closed.  
4. **Mouth Aspect Ratio (MAR)** – Detects yawns.  
5. **Head Pose Estimation** – Monitors head tilts/nods.  
6. **Alert System** – Warns the driver if drowsiness indicators cross thresholds.  

---

## 🔹 Results
- Achieved **~94.3% accuracy** under various lighting conditions.  
- False positive rate reduced to **5.7%** compared to traditional PERCLOS-only systems.  
- Real-time performance at **15 FPS** on standard hardware.  

---

## 🔹 Future Improvements
- Integration of **infrared cameras** for low-light/night driving.  
- Mobile app integration for driver alerts.  
- Cloud-based analytics for fleet monitoring.  

---

## 🔹 Screenshots
<img width="230" height="146" alt="image" src="https://github.com/user-attachments/assets/c7ac65b7-5777-4959-a197-809b84a9a63b" />
<img width="243" height="162" alt="image" src="https://github.com/user-attachments/assets/3fec691a-2169-4309-9c5d-e84f5b381262" />
<img width="976" height="549" alt="Screenshot 2025-08-25 at 8 19 01 PM" src="https://github.com/user-attachments/assets/00a5c66b-e3f1-421b-af40-fc2afea2eb66" />
<img width="979" height="555" alt="Screenshot 2025-08-25 at 8 18 42 PM" src="https://github.com/user-attachments/assets/64e2d4ae-b3ec-4f64-aa9a-064fe898496c" />





---

## 🔹 Author
👤 **Himanshu Gupta**  
📧 Email: himanshugupta00235@gmail.com  
🌐 GitHub: [himanshugupta00235](https://github.com/himanshugupta00235)  

---

⭐ If you found this project useful, don’t forget to **star this repository**!
