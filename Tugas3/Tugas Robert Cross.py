import cv2
import numpy as np

# Membaca gambar dari file
img = cv2.imread('C:/Users/Lenovo/Pictures/Camera Roll/WIN_20240625_08_15_09_Pro.jpg', 100)

# Kernel untuk operator Roberts Cross
roberts_cross_x = np.array([[1, 0], [0, -1]], dtype=np.float32)  # Kernel untuk arah X
roberts_cross_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)  # Kernel untuk arah Y

# Melakukan filter dengan kernel Roberts Cross
edge_x = cv2.filter2D(img, cv2.CV_64F, roberts_cross_x)
edge_y = cv2.filter2D(img, cv2.CV_64F, roberts_cross_y)

# Menghitung magnitudo gradien
edge_roberts = np.sqrt(np.square(edge_x) + np.square(edge_y))

# Mengonversi hasil ke format 8-bit untuk visualisasi
edge_roberts = cv2.convertScaleAbs(edge_roberts)

# Menampilkan gambar asli dan hasil deteksi tepi Roberts Cross
cv2.imshow('Gambar Asli', img)
cv2.imshow('Tepi Roberts Cross', edge_roberts)

# Menunggu hingga tombol ditekan dan menutup semua jendela
cv2.waitKey(0)
cv2.destroyAllWindows()