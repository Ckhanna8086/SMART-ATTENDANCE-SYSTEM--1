import face_recognition 
import cv2              
import os               
import pickle           
from imutils import paths 

print("[INFO] Encoding faces...")
imagePaths = list(paths.list_images("Test_Faces")) 
knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] Processing image {i+1}/{len(imagePaths)}")

    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
  
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model="cnn")
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

print("[INFO] Serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}

with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Face encodings saved to encodings.pickle")