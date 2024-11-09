import torch
import cv2
import numpy as np
from PIL import Image

# YOLOv5 modelini yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Görselinizi yükleyin
img_path = r"C:\Users\yunus\Desktop\goruntu_isleme\YOK\offside\Ekran görüntüsü 2024-10-24 101941.png"
img = Image.open(img_path)
results = model(img_path)

# Tespit edilen nesneleri alın
detections = results.xyxy[0]  # format [x_min, y_min, x_max, y_max, confidence, class]

# Görseli cv2 formatına çevir
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Renk paleti ve etiketler
team_colors = {
    "Team A": (255, 0, 0),    # Kırmızı
    "Team B": (0, 0, 255),    # Mavi
    "Goalkeeper": (0, 255, 0),  # Yeşil
    "Referee": (255, 255, 0)   # Sarı
}

# Renk yoğunluğu aralıkları (BGR)
color_ranges = {
    "Team A": ([0, 0, 150], [80, 80, 255]),  # Kırmızı
    "Team B": ([150, 0, 0], [255, 80, 80]),  # Mavi
    "Goalkeeper": ([0, 150, 0], [80, 255, 80]),  # Yeşil
    "Referee": ([0, 200, 200], [100, 255, 255])  # Sarı
}

# Sadece 'person' sınıfını filtreleyin
person_detections = []

for i, (*box, conf, cls) in enumerate(detections):
    if int(cls) == 0:  # 0 sınıfı 'person' olduğu için kontrol ediyoruz
        x_min, y_min, x_max, y_max = map(int, box)

        # Oyuncunun yüzdesinin etrafındaki alanı elde et
        player_area = img_cv[y_min:y_max, x_min:x_max]

        # Alanın ortalama rengini hesapla
        avg_color = cv2.mean(player_area)[:3]  # BGR formatında
        avg_color = np.array(avg_color, dtype=np.uint8)  # Renk değerini numpy dizisine çevir

        # Takım belirleme
        assigned_team = "Unknown"
        for team, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)  # BGR formatında numpy dizisi
            upper = np.array(upper, dtype=np.uint8)
            # Ortalama rengin belirtilen aralıklara uyup uymadığını kontrol et
            if np.all(lower <= avg_color) and np.all(avg_color <= upper):
                assigned_team = team
                break

        # Takım ve etiket yaz
        color = team_colors.get(assigned_team, (255, 255, 255))  # Bilinmeyen oyuncular için beyaz renk

        # Dikdörtgen çiz (oyuncunun etrafına kutu)
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), color, 2)

        # Etiket yazısı
        label = f"{assigned_team} ({conf:.2f})"
        cv2.putText(img_cv, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # "person" sınıfı olan tespitleri kaydet
        person_detections.append((x_min, y_min, x_max, y_max))

# Çizgileri tespit etmek için Canny edge detection kullanın
gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)

# Hough Line Transform ile çizgileri tespit et
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Tespit edilen çizgileri görüntü üzerine çizin
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Çizgiyi yeşil renkte çiz

# Sonucu göster
cv2.imshow("Categorized Players and Lines", img_cv)
cv2.waitKey(0)  # Herhangi bir tuşa basılana kadar bekle
cv2.destroyAllWindows()
