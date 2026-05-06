import cv2

print("Checking webcam...")
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Could not open webcam. Check if it's connected and not in use by another application.")
    exit()

print("Webcam opened successfully! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    cv2.imshow("Webcam Test - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam test finished.")
