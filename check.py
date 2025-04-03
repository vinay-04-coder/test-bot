from flask import Flask, request, jsonify, render_template
import cv2
import os
import pickle
import numpy as np
import face_recognition

app = Flask(__name__)

# Face recognition class
class LocalFaceRecognition:
    def __init__(self):
        self.dataset_path = "faces_dataset"
        self.encoding_file = "face_encodings.pkl"

        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        self.known_faces = []
        self.known_names = []

        self.load_known_faces()

    def save_encodings(self):
        with open(self.encoding_file, "wb") as file:
            pickle.dump((self.known_faces, self.known_names), file)
        print("Face encodings saved locally.")

    def load_known_faces(self):
        if os.path.exists(self.encoding_file):
            with open(self.encoding_file, "rb") as file:
                self.known_faces, self.known_names = pickle.load(file)
            print("Loaded stored face encodings.")
        else:
            print("No pre-stored encodings found.")

    def update_known_faces_from_dataset(self):
        for file_name in os.listdir(self.dataset_path):
            if file_name.endswith(".jpg"):
                image_path = os.path.join(self.dataset_path, file_name)
                name = os.path.splitext(file_name)[0]
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)

                if encoding:
                    self.known_faces.append(encoding[0])
                    self.known_names.append(name)
                    print(f"Loaded: {name}")

        self.save_encodings()

    def recognize_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_faces = []
        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_faces, encoding)
            name = "Unknown"

            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(self.known_faces, encoding))
                name = self.known_names[best_match_index]

            recognized_faces.append((name, (top, right, bottom, left), encoding))
        
        return recognized_faces

recognition = LocalFaceRecognition()

# Face Recognition API
@app.route("/detect_face", methods=["POST"])
def detect_face():
    cap = cv2.VideoCapture(0)

    try:
        ret, frame = cap.read()
        if not ret:
            return jsonify({"response": "Unable to capture image.", "speak": True})

        recognized_faces = recognition.recognize_faces(frame)

        for name, _, _ in recognized_faces:
            if name != "Unknown":
                return jsonify({"response": f"Welcome back, {name}!", "speak": True})
            else:
                return jsonify({"response": "Please enter your full name.", "speak": True, "new_user": True})

        return jsonify({"response": "No face detected.", "speak": True})

    finally:
        cap.release()

# Register New User API
@app.route("/register_user", methods=["POST"])
def register_user():
    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"response": "Name is required.", "speak": True})

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"response": "Unable to capture image.", "speak": True})

    # Save image locally
    file_path = os.path.join(recognition.dataset_path, f"{name}.jpg")
    cv2.imwrite(file_path, frame)
    print(f"Face registered as {name}.")

    # Encode face and update dataset
    face_encodings = face_recognition.face_encodings(frame)
    if not face_encodings:
        return jsonify({"response": "No face detected.", "speak": True})

    recognition.known_faces.append(face_encodings[0])
    recognition.known_names.append(name)
    recognition.save_encodings()

    return jsonify({"response": f"Welcome, {name}! You have been registered locally.", "speak": True})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
