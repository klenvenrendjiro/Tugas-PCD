# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 21:58:29 2024

@author: Lenovo
"""

import cv2
import numpy as np

# Membaca gambar dari file
img = cv2.imread('C:/Users/Lenovo/Pictures/Camera Roll/WIN_20240625_08_15_09_Pro.jpg', 100)


# Menggunakan operator Scharr untuk mendeteksi tepi
scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)  # Gradien pada arah X
scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)  # Gradien pada arah Y

# Menghitung magnitudo gradien (menggabungkan X dan Y)
scharr_combined = cv2.magnitude(scharr_x, scharr_y)

# Mengonversi hasil ke format 8-bit untuk visualisasi
scharr_x = cv2.convertScaleAbs(scharr_x)
scharr_y = cv2.convertScaleAbs(scharr_y)
scharr_combined = cv2.convertScaleAbs(scharr_combined)

# Menampilkan gambar asli dan hasil deteksi tepi Scharr
cv2.imshow('Gambar Asli', img)
cv2.imshow('Tepi Scharr X', scharr_x)
cv2.imshow('Tepi Scharr Y', scharr_y)
cv2.imshow('Tepi Scharr Gabungan', scharr_combined)

# Menunggu hingga tombol ditekan dan menutup semua jendela
cv2.waitKey(0)
cv2.destroyAllWindows()