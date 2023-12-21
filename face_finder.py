import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def save_faces(image_path, faces):
    image = cv2.imread(image_path)

    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]
        new_image_path = f"{os.path.splitext(image_path)[0]}_face_{i+1}.jpg"
        cv2.imwrite(new_image_path, face)
        print(f"Face saved as {new_image_path}")

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                faces = detect_faces(image_path)

                if len(faces) > 0:
                    save_faces(image_path, faces)
                os.remove(image_path)

# process_folder("dataset")
# process_folder("tests")
process_folder("target")