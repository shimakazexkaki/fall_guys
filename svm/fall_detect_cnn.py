import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# ----- 參數設定 -----
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 2

# ----- 自訂 Dataset -----
class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # 將標籤轉為 long 型別
        self.y = torch.tensor(y.values, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        # 原始資料為1D向量，改為 [1, 132] 以符合 conv1d 輸入 (channel=1)
        sample = self.X[idx].unsqueeze(0)
        label = self.y[idx]
        return sample, label

# ----- 定義 CNN 模型 -----
class FallDetectionCNN(nn.Module):
    def __init__(self, input_length, num_classes):
        super(FallDetectionCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        # 計算經過兩次池化後的特徵長度
        conv_output_length = input_length // 2 // 2  # input_length/4
        self.fc1 = nn.Linear(32 * conv_output_length, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x: (batch, 1, input_length)
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

def main():
    # ----- 讀取資料 -----
    df = pd.read_csv("../pose_train_data.csv")
    # 分離特徵與標籤
    X = df.drop("label", axis=1)
    y = df["label"]
    input_length = X.shape[1]  # 例如 33*4 = 132

    # 標準化處理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 切分訓練與測試集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 建立 Dataset 與 DataLoader
    train_dataset = PoseDataset(X_train, y_train)
    test_dataset = PoseDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ----- 建立 CNN 模型 -----
    model = FallDetectionCNN(input_length=input_length, num_classes=NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ----- 訓練模型 -----
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # (batch, 1, input_length)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # (batch, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")
    
    # ----- 評估模型 -----
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    print("評估報告：")
    print(classification_report(all_labels, all_preds, target_names=["Not Fallen", "Fallen"]))
    
    # ----- 儲存模型與標準化器 -----
    torch.save(model.state_dict(), "cnn_fall_model.pt")
    joblib.dump(scaler, "cnn_scaler.pkl")
    print("CNN 模型與標準化器已儲存")

if __name__ == "__main__":
    main()