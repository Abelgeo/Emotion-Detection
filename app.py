from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load the trained model
model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load('emotion_model.pth', map_location=torch.device('cpu')))
model.eval()
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    data = request.json['frame']  # base64-encoded frame
    img_data = base64.b64decode(data.split(',')[1])
    image = Image.open(io.BytesIO(img_data))
    img_array = np.array(image)

    # Detect faces
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return jsonify({'emotion': 'no_face', 'confidence': 0.0})

    # Process the first face
    (x, y, w, h) = faces[0]
    face = img_array[y:y+h, x:x+w]
    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    # Transform and predict
    input_tensor = transform(face_pil).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 0)
        emotion = emotions[predicted.item()]

    return jsonify({'emotion': emotion, 'confidence': confidence.item()})

if __name__ == '__main__':
    app.run(debug=True)