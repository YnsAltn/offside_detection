import torch
import cv2
import numpy as np
from PIL import Image

# YOLOv5 modelini yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Görselinizi yükleyin
img_path = "C:/Users/yunus/Desktop/goruntu_isleme/YOK/offside/Ekran görüntüsü 2024-10-24 101941.png"
img = Image.open(img_path)
results = model(img_path)

# Tespit edilen nesneleri alın
detections = results.xyxy[0]  # format [x_min, y_min, x_max, y_max, confidence, class]

# Görseli cv2 formatına çevir
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Sadece 'person' sınıfını filtreleyin ve her biri için etiket ekleyin
person_detections = []  # "person" sınıfı olan nesneleri kaydetmek için liste

for i, (*box, conf, cls) in enumerate(detections):
    if int(cls) == 0:  # 0 sınıfı 'person' olduğu için kontrol ediyoruz
        x_min, y_min, x_max, y_max = map(int, box)

        # Dikdörtgen çiz (oyuncunun etrafına kutu)
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Etiket yazısı ve güven puanı
        label = f"person {i + 1} ({conf:.2f})"

        # Etiketi yaz (oyuncunun üstüne "person X" formatında)
        cv2.putText(img_cv, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # "person" sınıfı olan tespitleri kaydet
        person_detections.append((x_min, y_min, x_max, y_max))

# En sağdaki savunma oyuncusunun x_max değerine göre ofsayt çizgisi çiz
if person_detections:
    # x_max değeri en büyük olan oyuncuyu bul
    max_x_person = max(person_detections, key=lambda x: x[2])  # x_max'e göre en sağdaki oyuncu
    offside_line_x = int(max_x_person[2])  # Bu oyuncunun x_max değeri çizgi için kullanılacak

    # Çizginin nerede olduğunu görmek için kalınlığını 5 piksel yap
    # Çizginin görselde yer alıp almadığını kontrol et
    if 0 <= offside_line_x < img_cv.shape[1]:
        cv2.line(img_cv, (offside_line_x, 0), (offside_line_x, img_cv.shape[0]), (0, 0, 255), 5)
    else:
        print(f"Çizgi çizmek için hesaplanan değer geçersiz: {offside_line_x}")

# Sonucu göster
cv2.imshow("Offside Line with Detected Persons", img_cv)
cv2.waitKey(0)  # Herhangi bir tuşa basılana kadar bekle
cv2.destroyAllWindows()
cv2.imwrite("output_image.png", img_cv)
