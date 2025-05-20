from flask import Flask, Response
import cv2
import mediapipe as mp
import joblib
import numpy as np
import torch
import torch.nn as nn

# 定義 CNN 模型 (與訓練時保持一致)
class FallDetectionCNN(nn.Module):
    def __init__(self, input_length, num_classes):
        super(FallDetectionCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        conv_output_length = input_length // 2 // 2  # 經過兩層池化 (kernel_size=2)
        self.fc1 = nn.Linear(32 * conv_output_length, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch, 1, input_length)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch, 16, input_length/2)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch, 32, input_length/4)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化 Flask
app = Flask(__name__)

# 初始化攝影機
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 載入 CNN 模型與標準化器
INPUT_LENGTH = 132  # 33個關鍵點，每個點4個數據 (x, y, z, visibility)
NUM_CLASSES = 2
model = FallDetectionCNN(INPUT_LENGTH, NUM_CLASSES)
model.load_state_dict(torch.load("cnn_fall_model.pt", map_location=torch.device("cpu")))
model.eval()

scaler = joblib.load("cnn_scaler.pkl")

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 每幀處理與推論
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        status_text = "Unknown"
        color = (128, 128, 128)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

            if len(keypoints) == INPUT_LENGTH:
                # 使用標準化器與 CNN 模型進行推論
                X_input = scaler.transform([keypoints])
                # 調整成符合 CNN 輸入格式: (batch, channel, input_length)
                X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(1)
                with torch.no_grad():
                    outputs = model(X_tensor)
                    prediction = torch.argmax(outputs, dim=1).item()
                if prediction == 1:
                    status_text = "FALLEN"
                    color = (0, 0, 255)
                else:
                    status_text = "NOT FALLEN"
                    color = (0, 255, 0)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        # 傳送到瀏覽器
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# 網頁首頁
@app.route('/')
def index():
    return '''
        <html><body>
        <h1>即時跌倒偵測 - CNN 模型版</h1>
        <img src="/video_feed">
        </body></html>
    '''

# 串流路由
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 執行伺服器
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)