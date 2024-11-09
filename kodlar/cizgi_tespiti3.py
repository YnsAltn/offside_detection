import torch
import cv2
import numpy as np

# YOLOv5 modelini yükleyin
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' küçük ve hızlı model

# Görselin yolu
img_path = "C:/Users/yunus/Desktop/goruntu_isleme/YOK/offside/Ekran görüntüsü 2024-10-24 101941.png"

# Görseli yükleyin
img = cv2.imread(img_path)

# YOLOv5 ile görseldeki nesneleri tespit edin
results = model(img)

# Tespit edilen sonuçları pandas dataframe olarak alıyoruz
df = results.pandas().xywh[0]  # İlk dataframe'i alıyoruz

# 'class' sütununa göre sadece 'person' tespitlerini seçiyoruz (0: person sınıfı)
players = df[df['class'] == 0]  # 0, 'person' sınıfının ID'sidir

# Tespit edilen oyuncuları görsel üzerinde işaretleyin
for index, row in players.iterrows():
    # Merkez koordinatları ve boyutlar
    xcenter, ycenter, width, height = row['xcenter'], row['ycenter'], row['width'], row['height']

    # Bounding box hesaplama
    xmin = int(xcenter - width / 2)
    ymin = int(ycenter - height / 2)
    xmax = int(xcenter + width / 2)
    ymax = int(ycenter + height / 2)

    # Oyuncuyu işaretleme
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Yeşil renk ile dikdörtgen çiz

# Görseli gri tonlamaya çevir
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny kenar tespiti
edges = cv2.Canny(gray, 50, 150)

# Hough dönüşümü ile çizgileri tespit etme
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100)

# Çizgileri görselleştirme
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Kırmızı renk ile çizgi çiz

# En ön oyuncunun (en küçük x koordinatına sahip olan) tespit edilmesi
min_x = min(players['xcenter'])
offside_player = players[players['xcenter'] == min_x].iloc[0]

# Ofsayt çizgisi için belirlenen eğimi çizeceğiz
# Bu çizgi, oyuncunun en yakın çim çizgisine paralel olacak
# Ofsayt çizgisinin başlangıç ve bitiş noktalarını hesaplayın (bu adımda basitleştirilmiş bir çizim yapılacak)
x1, y1 = int(offside_player['xcenter'] - offside_player['width'] / 2), int(
    offside_player['ycenter'] - offside_player['height'] / 2)
x2, y2 = x1 + 200, y1  # Basit bir yatay çizgi ekleme (geliştirilebilir)

# Ofsayt çizgisini çizme
cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Mavi renk ile çizgi çiz

# Sonucu kaydet
output_path = r"C:\Users\yunus\Documents\GitHub\offside_detection\kodlar\final_offside_result.png"
cv2.imwrite(output_path, img)
print(f"Sonuç kaydedildi: {output_path}")


# Sonuçları görselleştirme
cv2.imshow("Offside Line", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
