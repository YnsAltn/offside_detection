import torch
import cv2
import numpy as np
from sklearn.cluster import KMeans

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


# Renk sınıflandırma için RGB ortalamalarını almak
def get_average_color(image, bbox):
    """
    Verilen bounding box içinde bir resmin ortalama renk değerini döndürür.
    """
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]  # ROI (Region of Interest) seçimi
    average_color = np.mean(roi, axis=(0, 1))  # Ortalama R, G, B değeri
    return average_color


# Oyuncuları renklerine göre sınıflandırma
player_colors = []

# Her bir oyuncunun rengini hesaplayalım
for index, row in players.iterrows():
    xcenter, ycenter, width, height = row['xcenter'], row['ycenter'], row['width'], row['height']

    # Bounding box hesaplama
    xmin = int(xcenter - width / 2)
    ymin = int(ycenter - height / 2)
    xmax = int(xcenter + width / 2)
    ymax = int(ycenter + height / 2)

    # Renk analizini yapıyoruz
    avg_color = get_average_color(img, (xmin, ymin, xmax, ymax))
    player_colors.append(avg_color)

# Renkleri k-means ile sınıflandırıyoruz (örneğin 3 renkten fazla sınıf oluşturulabilir)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(player_colors)

# Kümelere göre oyuncuları sınıflandırıyoruz
players['color_cluster'] = kmeans.labels_

# Görseldeki oyuncuları renklerine göre sınıflandırarak işaretleme
for index, row in players.iterrows():
    xcenter, ycenter, width, height = row['xcenter'], row['ycenter'], row['width'], row['height']

    # Bounding box hesaplama
    xmin = int(xcenter - width / 2)
    ymin = int(ycenter - height / 2)
    xmax = int(xcenter + width / 2)
    ymax = int(ycenter + height / 2)

    # Renk kümesini alıyoruz
    color_cluster = row['color_cluster']

    # Renk kümesine göre sınıf rengi belirliyoruz (örneğin 0: kırmızı, 1: mavi, 2: yeşil)
    if color_cluster == 0:
        color = (0, 0, 255)  # Kırmızı
    elif color_cluster == 1:
        color = (255, 0, 0)  # Mavi
    else:
        color = (0, 255, 0)  # Yeşil

    # Oyuncuyu işaretleme
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)  # İlgili renk ile dikdörtgen çiz

# Çizgileri tespit etmek için Hough Line Transform kullanma
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Çizgileri tespit et
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Çizgileri görsele ekleme
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Sarı çizgi


# Ofsayt çizgisi için en soldaki oyuncuyu bulalım
# Bu oyuncu, en küçük 'x' koordinatına sahip olan oyuncu olacak
def find_defensive_player(players):
    """
    Savunma oyuncusunun yerini bulmak için en soldaki oyuncuyu seçiyoruz.
    """
    return players.iloc[players['xcenter'].idxmin()]


def draw_offside_line(img, defensive_player, offset=100):
    """
    Savunma oyuncusunun yerini ve ofsayt çizgisini görsele çizen fonksiyon.
    """
    # Ofsayt çizgisi için savunma oyuncusunun xcenter'ini alıyoruz
    offside_line_x = int(defensive_player['xcenter']) + offset  # 100 px ileriye çizmek
    cv2.line(img, (offside_line_x, 0), (offside_line_x, img.shape[0]), (255, 0, 0), 3)  # Mavi ofsayt çizgisi


# En soldaki (savunma) oyuncuyu buluyoruz
defensive_player = find_defensive_player(players)

# Ofsayt çizgisini çizeceğiz
draw_offside_line(img, defensive_player, offset=100)

# Sonucu kaydet
output_path = r"C:\Users\yunus\Documents\GitHub\offside_detection\kodlar\final_offside_result.png"
cv2.imwrite(output_path, img)
print(f"Sonuç kaydedildi: {output_path}")

# Sonuçları görselleştirme
cv2.imshow("Classified Players with Lines and Offside", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
