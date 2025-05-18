## 樹梅派安裝一些套件
sudo apt update  
sudo apt install python3-opencv python3-flask python3-pip  
pip3 install mediapipe --break-system-packages  
pip3 install scikit-learn --break-system-packages  
  
  
### 在自己的電腦訓練svm
pip install mediapipe opencv-python scikit-learn matplotlib pandas  
  
#### 跌倒判斷訓練資料(kaggle)  
https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset?resource=download  

#### mediapipe
https://github.com/google-ai-edge/mediapipe?tab=readme-ov-file

在做mediapipe label標記的時候要先對圖片做預處理，改成跟相機拍下來同個解析度。
鏡頭調整成640*480。
