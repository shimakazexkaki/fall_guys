import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 讀取 pose_train_data.csv
df = pd.read_csv("../pose_train_data.csv")

# 分離特徵與標籤
X = df.drop("label", axis=1)
y = df["label"]

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 切分訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 建立 SVM 模型（RBF kernel 最常用）
svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_train, y_train)

# 測試模型
y_pred = svm.predict(X_test)

# 印出結果
print("評估報告：")
print(classification_report(y_test, y_pred, target_names=["Not Fallen", "Fallen"]))
# 儲存模型與標準化器
joblib.dump(svm, "svm_fall_model.pkl")
joblib.dump(scaler, "svm_scaler.pkl")
print("模型與標準化已儲存")
