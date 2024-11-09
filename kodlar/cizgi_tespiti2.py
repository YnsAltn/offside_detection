import torch
import cv2
import numpy as np
from PIL import Image

# YOLOv5 modelini yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Görseli yükle
img_path = "C:/Users/yunus/Desktop/goruntu_isleme/YOK/offside/Ekran görüntüsü 2024-10-24 101941.png"
img = Image.open(img_path)
results = model(img_path)

# Tespit edilen nesneleri alın
detections = results.xyxy[0]

# Görseli cv2 formatına çevir
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Sadece 'person' sınıfını filtreleyin
person_detections = []
for i, (*box, conf, cls) in enumerate(detections):
    if int(cls) == 0:  # Sınıf 0, yani "person"
        x_min, y_min, x_max, y_max = map(int, box)
        person_detections.append((x_min, y_min, x_max, y_max))

# En soldaki oyuncuyu tespit et (savunma oyuncusu olarak varsayıyoruz)
if person_detections:
    leftmost_player = min(person_detections, key=lambda b: b[0])
    x_min, y_min, x_max, y_max = leftmost_player
    player_center_x = (x_min + x_max) // 2  # Oyuncunun merkez noktası (x ekseninde)

    # Görüntüyü gri tonlamaya çevir ve kenar tespiti yap
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)

    # Çim çizgileriyle paralel olan çizgileri bul
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Çim çizgilerine paralel olan en dik çizgiyi bul
    if lines is not None:
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Dik çizgiler için eğim kontrolü: x farkı çok küçükse
            if abs(x1 - x2) < 10:
                vertical_lines.append((x1, y1, x2, y2))

        # En yakın dikey çizgiyi referans olarak seç
        if vertical_lines:
            reference_line = min(vertical_lines, key=lambda line: abs(line[0] - player_center_x))
            x1, y1, x2, y2 = reference_line
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Çizginin eğimi

            # Oyuncunun merkez noktasından başlatılan eğimli ofsayt çizgisi
            offset_x = int(player_center_x)  # Tam sayı olarak dönüşüm
            offset_y1 = int(y1 + slope * (offset_x - x1))  # Tam sayı olarak dönüşüm
            offset_y2 = int(y2 + slope * (offset_x - x2))  # Tam sayı olarak dönüşüm

            # Çizgiyi kırmızı renkle çiz
            cv2.line(img_cv, (offset_x, offset_y1), (offset_x, offset_y2), (0, 0, 255), 2)

# Sonucu kaydet
output_path = r"C:\Users\yunus\Documents\GitHub\offside_detection\kodlar\final_offside_result.png"
cv2.imwrite(output_path, img_cv)
print(f"Sonuç kaydedildi: {output_path}")
