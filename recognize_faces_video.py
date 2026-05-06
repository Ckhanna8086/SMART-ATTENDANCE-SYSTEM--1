import face_recognition
import cv2
import os
import pickle
from datetime import datetime
import pandas as pd
import time 

print("[INFO] Loading encodings...")
try:
    with open("encodings.pickle", "rb") as f:
        data = pickle.loads(f.read())
except FileNotFoundError:
    print("[ERROR] encodings.pickle not found. Please run encode_faces.py first.")
    exit()


print("[INFO] Starting video stream...")
vs = cv2.VideoCapture(0) 


attendance_file = "attendance.csv"
if os.path.exists(attendance_file):
    df_attendance = pd.read_csv(attendance_file)
    print(f"[INFO] Loaded existing attendance from {attendance_file}")
else:

    df_attendance = pd.DataFrame(columns=["Name", "Date", "Time"])
    print(f"[INFO] Created new attendance file: {attendance_file}")

recognized_today = {}

print("[INFO] Attendance system is running. Press 'q' to quit.")

while True:

    ret, frame = vs.read()
    if not ret:
        print("[ERROR] Failed to grab frame from webcam. Exiting...")
        break
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb_small_frame, model="cnn")
    encodings = face_recognition.face_encodings(rgb_small_frame, boxes)
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
        name = "Unknown" 
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
            now = datetime.now()
            today_date = now.strftime("%Y-%m-%d")
            current_time = now.strftime("%H:%M:%S")
            if name not in recognized_today or (now - recognized_today[name]).total_seconds() > 300:
                new_entry = pd.DataFrame([{"Name": name, "Date": today_date, "Time": current_time}])
                df_attendance = pd.concat([df_attendance, new_entry], ignore_index=True)
                df_attendance.to_csv(attendance_file, index=False)
                print(f"[ATTENDANCE] {name} recorded at {current_time} on {today_date}")
                recognized_today[name] = now
        names.append(name)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        top = int(top * 2)
        right = int(right * 2)
        bottom = int(bottom * 2)
        left = int(left * 2)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) 
        text_color = (0, 0, 0) if name != "Unknown" else (255, 255, 255) 
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    cv2.imshow("AI-Powered Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vs.release()
cv2.destroyAllWindows()
print("[INFO] Attendance system stopped.")