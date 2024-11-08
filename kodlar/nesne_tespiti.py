import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# YOLOv5'ten mevcut modellerden birini yükleyin, örneğin 'yolov5s' (small model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Oyuncuları belirlemek için kişiyi temsil eden 'person' sınıfını filtrelemek amacıyla
# modelin sadece "person" tespitlerini seçmesi sağlanır.

# Görselinizi yükleyin
img_path = "C:/Users/yunus/Desktop/Görüntü İşleme/YOK/offside/Ekran görüntüsü 2024-10-24 024654.png"

# Modeli görsel üzerinde çalıştırın
results = model(img_path)

# Tespit edilen nesneleri görselleştir
results.show()

# Tespit sonuçlarını görüntülemek için koordinatları ve sınıfları alabilirsiniz
detections = results.xyxy[0]  # format [x_min, y_min, x_max, y_max, confidence, class]
for *box, conf, cls in detections:
    if int(cls) == 0:  # 'person' sınıfı YOLO'da genellikle 0 ile ifade edilir
        print(f"Koordinatlar: {box}, Güven: {conf}, Sınıf: {cls}")
