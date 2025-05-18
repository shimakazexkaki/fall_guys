import os
import csv
import cv2
import mediapipe as mp

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# 開啟輸出檔
output_csv = open("pose_train_data.csv", mode='w', newline='')
writer = csv.writer(output_csv)

# 寫入標題
header = []
for i in range(33):
    header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
header.append("label")
writer.writerow(header)

# 處理 train 資料夾下的所有圖片
base_dir = "./fall_dataset/images/train"
for filename in os.listdir(base_dir):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    # 根據檔名決定標籤
    if filename.startswith("fall"):
        label = 1
    elif filename.startswith("not fallen"):
        label = 0
    else:
        print(f"檔名有錯誤：{filename}")
        continue

    img_path = os.path.join(base_dir, filename)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (640, 480))
    if image is None:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        row = []
        for lm in result.pose_landmarks.landmark:
            row += [lm.x, lm.y, lm.z, lm.visibility]
        row.append(label)
        writer.writerow(row)
    else:
        print(f"沒有偵測到姿勢：{filename}")

output_csv.close()
print("pose_train_data.csv 完成！")