import cv2
import numpy as np
from matplotlib import pyplot as plt

img_color = cv2.imread('C:/Users/KLENVEN/Documents/pengolahan cinta/misel.jpg', 1)
img_gray = cv2.imread('C:/Users/KLENVEN/Documents/pengolahan cinta/misel.jpg', 0)


histr_img = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
filter_maximum = cv2.dilate(img_gray, np.ones((3, 3), np.uint8))
filter_minimum = cv2.erode(img_gray, np.ones((3, 3), np.uint8))
filter_median = cv2.medianBlur(img_gray, 3)
filter_gaussian = cv2.GaussianBlur(img_gray, (3, 3), 0)

histr_maximum = cv2.calcHist([filter_maximum], [0], None, [256], [0, 256])
histr_minimum = cv2.calcHist([filter_minimum], [0], None, [256], [0, 256])
histr_median = cv2.calcHist([filter_median], [0], None, [256], [0, 256])
histr_gaussian = cv2.calcHist([filter_gaussian], [0], None, [256], [0, 256])

titles = ['Original Image', 'Maximum Filter', 'Minimum Filter', 'Median Filter', 'Gaussian Filter']
images = [img_color, filter_maximum, filter_minimum, filter_median, filter_gaussian]
histograms = [histr_img, histr_maximum, histr_minimum, histr_median, histr_gaussian]

plt.figure(figsize=(12, 10))
for i in range(5):
    plt.subplot(5, 2, 2*i+1)
    if i == 0:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))  # Mengonversi dari BGR ke RGB untuk tampilan yang benar
    else:
        plt.imshow(images[i], cmap='gray')
    
    plt.title(titles[i])
    plt.axis('off') 

    plt.subplot(5, 2, 2*i+2)
    plt.plot(histograms[i])
    plt.xlim([0, 256])
    plt.title(f"Histogram of {titles[i]}")
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()