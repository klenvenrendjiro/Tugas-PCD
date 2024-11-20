import cv2
import numpy as np
import matplotlib.pyplot as plt


img_color = cv2.imread('C:/Users/KLENVEN/Documents/pengolahan cinta/misel.jpg', 1)
img_gray = cv2.imread('C:/Users/KLENVEN/Documents/pengolahan cinta/misel.jpg', 0)


blur1 = cv2.blur(img_gray, (3, 3))
blur2 = cv2.GaussianBlur(img_gray, (3, 3), 0)
median = cv2.medianBlur(img_gray, 3)
blur3 = cv2.bilateralFilter(img_gray, 9, 75, 75)


def plot_image_and_histogram(image, title, is_color=False):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if is_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
        
    plt.title(f'{title} Image', fontsize=14)
    plt.axis('off')  


    plt.subplot(1, 2, 2)
    plt.hist(image.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title(f"Histogram of {title}", fontsize=14)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()


plot_image_and_histogram(img_color, "Original", is_color=True)
plot_image_and_histogram(blur1, "Averaging Blur")
plot_image_and_histogram(blur2, "Gaussian Blur")
plot_image_and_histogram(blur3, "Bilateral Blur")
plot_image_and_histogram(median, "Median Blur")

cv2.waitKey(0)
cv2.destroyAllWindows()