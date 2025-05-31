import cv2
import os

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory where the face data will be stored
data_dir = 'face rec/face_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def capture_faces(name):
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y + h, x:x + w]
            file_name_path = os.path.join(data_dir, f'{name}_{count}.jpg')
            cv2.imwrite(file_name_path, face)
            cv2.putText(frame, str(count), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face Capture', frame)

        if cv2.waitKey(1) == 27 or count >= 100:  # Escape key or 100 samples
            break

    cap.release()
    cv2.destroyAllWindows()

name = input("Enter your name: ")
capture_faces(name)
