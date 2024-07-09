from flask import Flask, render_template, Response
import cv2
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load OpenCV's pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the webcam
camera = cv2.VideoCapture(0)

def get_dominant_color(image, k=4):
    # Convert the image to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    # Use KMeans clustering to find the dominant colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    # Get the cluster centers (the dominant colors)
    colors = kmeans.cluster_centers_.astype(int)
    # Get the most common color (the dominant color)
    dominant_color = colors[kmeans.labels_[0]]
    return tuple(dominant_color)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                # Draw a rectangle around each face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Extract the face region
                face = frame[y:y+h, x:x+w]

                # Get the dominant color of the face region
                dominant_color = get_dominant_color(face)
                # Create a rectangle with the dominant color
                cv2.rectangle(frame, (x, y-30), (x+w, y), dominant_color, -1)
                # Put the dominant color as text
                cv2.putText(frame, f'Color: {dominant_color}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
