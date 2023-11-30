import cv2
from flask.json import jsonify
import pytesseract
from flask import Flask, render_template, request, Response
import numpy as np

app = Flask(__name__)

# Load the pre-trained cascade classifier for license plates
carPlatesCascade = cv2.CascadeClassifier('indian_license_plate.xml')

if carPlatesCascade.empty():
    print("Error loading cascade classifier!")
else:
    print("Cascade classifier loaded successfully!")

# Set up Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to process the frame and detect license plates
def detect_plates(frame):
    detected_plates = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car_plates = carPlatesCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(25, 25))

    for (x, y, w, h) in car_plates:
        plate = frame[y:y + h, x:x + w]
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        _, plate_threshold = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            plate_text = pytesseract.image_to_string(plate_threshold, config='--psm 8')
            plate_text = plate_text.strip()

            confidence_measure = min(1.0, max(0.0, len(plate_text) / 10))

            if plate_text and confidence_measure > 0.1:
                detected_plates.append({
                    "Plate": plate_text,
                    "Detected_Text": plate_text,
                    "Confidence": confidence_measure
                })

        except Exception as e:
            print(f"An error occurred: {e}")
            confidence_measure = 0.0

    return detected_plates

def video_stream():
    video_path = 'path_of_the__video.mp4' 
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detected_plates= detect_plates(frame)
        
        # Filter plates based on confidence (> 0.75) and print
        for plate_info in detected_plates:
            if plate_info['Confidence'] > 0.75:
                print(f"Plate: {plate_info['Plate']} - Confidence: {plate_info['Confidence']:.2f}")
        
        # Extend the list with plates having confidence > 0.75
        detected_plates.extend([plate_info for plate_info in detected_plates if plate_info['Confidence'] > 0.75])
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video provided"}), 400

    video_file = request.files['video']
    video_path = './static/uploaded_video.mp4'  
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)

    detected_plates = []  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect plates in the frame
        plates_in_frame = detect_plates(frame)
        detected_plates.extend(plates_in_frame) 


    cap.release()

    return render_template('result.html',video_path='/static/uploaded_video.mp4', detected_plates=detected_plates)


if __name__ == '__main__':
    app.run(debug=True)
