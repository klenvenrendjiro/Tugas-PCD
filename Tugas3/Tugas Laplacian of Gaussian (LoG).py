# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 23:34:18 2024

@author: Lenovo
"""

import cv2
import numpy as np

def LoG(img, sigma):
  # Membuat filter Gaussian
  size = 2*int(3*sigma) + 1
  x, y = np.meshgrid(np.arange(-size//2 + 1, size//2 + 1),
                     np.arange(-size//2 + 1, size//2 + 1))
  gaussian = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))

  # Membuat filter Laplacian
  laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)

  # Konvolusi
  img_filtered = cv2.filter2D(img, cv2.CV_64F, laplacian)

  # Ambang batas (adjust sesuai kebutuhan)
  thresh = np.mean(img_filtered)
  img_binary = np.where(img_filtered > thresh, 255, 0).astype(np.uint8)

  return img_binary

# Membaca citra grayscale
img = cv2.imread('C:/Users/Lenovo/Pictures/Camera Roll/WIN_20240625_08_15_09_Pro.jpg', 100)

# Menentukan nilai sigma
sigma = 2

# Menerapkan filter LoG
result = LoG(img, sigma)

# Menampilkan hasil
cv2.imshow('Original', img)
cv2.imshow('LoG', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
