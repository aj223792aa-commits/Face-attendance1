import base64
import os
from datetime import datetime
import cv2
import face_recognition
from flask import Flask, jsonify, render_template, request
import numpy as np

app = Flask(__name__)

# بارکردنی وێنەی کارمەندەکان (پێویستە وێنەکانیان لە بوخچەی known_faces بێت)
# ناوی وێنەکە دەبێت ناوی کارمەندەکە بێت، بۆ نموونە: "Ali_Karim.jpg"
KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []

if os.path.exists(KNOWN_FACES_DIR):
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]

            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan():
    data = request.json
    image_data = data["image"].split(",")[1]

    # گۆڕینی وێنەی وەرگیراو بۆ فرماتی ڕێگەپێدراو
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # دۆزینەوەی ڕوخسار لە وێنەکەدا
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=0.5
        )
        name = "نەنژێنراوە"

        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # تۆمارکردنی کات و بەرواری دەوام
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("attendance.csv", "a") as f:
                    f.write(f"{name},{now}\n")

                return jsonify(
                    {
                        "status": "success",
                        "message": f"بەرێز {name}، پەنجەمۆرەکەت لە کاتی {now} تۆمارکرا.",
                    }
                )

    return jsonify({"status": "fail", "message": "ڕوخسارەکەت نەناسرایەوە!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
