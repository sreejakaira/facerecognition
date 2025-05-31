Face Recognition and Machine Learning–Integrated Smart Attendance Management System for Educational Institutions
A smart, secure, and automated attendance tracking system using facial recognition and machine learning, built to modernize educational administration. This solution minimizes manual effort, eliminates proxy attendance, and ensures accurate, real-time logging.

🚀 Features
✅ Automated Attendance Marking

🧠 ML-Powered Facial Recognition

📷 Real-Time Camera-Based Detection

📊 Daily CSV Attendance Logs

🔒 Secure & Tamper-Resistant Records

⚙️ Scalable for Institutions of All Sizes

🏗️ Project Architecture
This system is implemented in two simple phases, each executable directly from Python IDLE:

1️⃣ training.py – Face Encoding (Run Once)
What it does:
Scans the dataset/ folder for student images.
Converts each face into a machine-readable encoding.
Saves all encodings into the trained_model/ directory.

How to run:
Open training.py in Python IDLE and press F5 to execute.

2️⃣ attend.py – Real-Time Attendance (Run Daily)
What it does:
Activates webcam, detects faces, compares with saved encodings, and updates attendance.csv in real-time.

How to run:
Open attend.py in Python IDLE and press F5.
Ensure proper lighting and the camera is enabled.

