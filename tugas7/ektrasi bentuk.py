import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to process and display each step
def process_image(image_path):
    # Load the image
    RGB = cv2.imread(image_path)
    RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    plt.figure()
    plt.imshow(RGB)
    plt.title("Original RGB Image")
    plt.show()

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(RGB, cv2.COLOR_RGB2GRAY)
    plt.figure()
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.show()

    # Step 2: Apply thresholding (using Otsu's method)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plt.figure()
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Image (Thresholded)")
    plt.show()

    # Step 3: Remove small objects by area thresholding
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_size = 30  # Minimum area threshold
    bw_filtered = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            bw_filtered[labels == i] = 255
    plt.figure()
    plt.imshow(bw_filtered, cmap='gray')
    plt.title("After Removing Small Objects")
    plt.show()

    # Step 4: Morphological closing
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw_closed = cv2.morphologyEx(bw_filtered, cv2.MORPH_CLOSE, se)
    plt.figure()
    plt.imshow(bw_closed, cmap='gray')
    plt.title("After Morphological Closing")
    plt.show()

    # Step 5: Fill holes using contours
    contours, _ = cv2.findContours(bw_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bw_filled = np.zeros_like(bw_closed)

    # Fill each contour
    for contour in contours:
        cv2.drawContours(bw_filled, [contour], -1, (255), thickness=cv2.FILLED)

    # Display the filled binary image
    plt.figure()
    plt.imshow(bw_filled, cmap='gray')
    plt.title("After Filling Holes")
    plt.show()

    # Step 6: Find contours and label shapes
    contours, _ = cv2.findContours(bw_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labeled_image = RGB.copy()

    # Calculate properties and label shapes
    for k, contour in enumerate(contours, start=1):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        metric = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Compute centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Display results in terminal
        print("===================================")
        print(f"Object number = {k}")
        print(f"Area = {area}")
        print(f"Perimeter = {perimeter}")
        print(f"Metric = {metric}")

        # Labeling the object
        label_text = f"{k} Bulat" if metric > 0.8 else f"{k} Tidak Bulat "
        cv2.putText(labeled_image, label_text, (cX - 10, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 0, 0), 1,6)
        cv2.drawContours(labeled_image, [contour], -10, (0, 128, 0), 2)

    # Display labeled image
    plt.figure()
    plt.imshow(labeled_image)
    plt.title("Labeled Image")
    plt.show()

# Process the image
process_image('C:/Users/Lenovo/Pictures/images.png')
