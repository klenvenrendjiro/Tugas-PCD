import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk memproses citra
def process_image(file_path):
    # Load citra dalam grayscale
    I = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(6, 6))
    plt.imshow(I, cmap='gray')
    plt.title("Citra Grayscale")
    plt.axis('off')
    plt.show()

    # Konversi ke citra biner menggunakan threshold Otsu
    _, bw = cv2.threshold(I, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure(figsize=(6, 6))
    plt.imshow(bw, cmap='gray')
    plt.title("Citra Biner")
    plt.axis('off')
    plt.show()

    # Operasi morfologi untuk mengisi lubang
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    plt.figure(figsize=(6, 6))
    plt.imshow(bw2, cmap='gray')
    plt.title("Citra Setelah Morfologi")
    plt.axis('off')
    plt.show()

    # Menemukan kontur
    contours, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Menampilkan hasil centroid dan labeling
    plt.figure(figsize=(10, 10))
    plt.imshow(I, cmap='gray')
    plt.title("Hasil Labeling dan Centroid")
    plt.axis('off')

    for k, contour in enumerate(contours, start=1):
        # Menghitung area dan perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Menghitung centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Plot centroid
        plt.plot(cX, cY, 'r*')

        # Plot kontur
        contour_array = np.array(contour).reshape((-1, 2))
        plt.plot(contour_array[:, 0], contour_array[:, 1], 'y', linewidth=2)

        # Menampilkan teks label dan properti
        plt.text(cX, cY - 10, f'Label = {k}', color='red', fontsize=12, fontweight='bold')
        plt.text(cX, cY, f'Area = {area:.2f}', color='blue', fontsize=12, fontweight='bold')
        plt.text(cX, cY + 10, f'Perim = {perimeter:.2f}', color='green', fontsize=12, fontweight='bold')
        plt.text(cX, cY + 20, f'X = {cX:.2f}', color='cyan', fontsize=12, fontweight='bold')
        plt.text(cX, cY + 30, f'Y = {cY:.2f}', color='cyan', fontsize=12, fontweight='bold')

    plt.show()

# Path gambar yang ingin diproses
file_path = 'C:/Users/Lenovo/Pictures/o.png'
process_image(file_path)
