import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance

# Function to read and process DICOM image
def process_dicom_image(dicom_filename):
    try:
        # Read DICOM image
        ds = pydicom.dcmread(dicom_filename, force=True)
        image = ds.pixel_array

        # Show original DICOM image
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title('Original DICOM Image')
        plt.axis('off')
        plt.show()

        # Define a threshold value to isolate white objects
        threshold_value = 200  # Adjust this value as needed for your specific image
        binary_image = (image > threshold_value).astype(np.uint8) * 255  # Create binary image

        # Show binary image
        plt.figure()
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binary Image (White Objects)')
        plt.axis('off')
        plt.show()

        # Find contours and calculate centroids
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f'Number of contours found: {len(contours)}')

        centroids = []

        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                centroids.append((cX, cY))

        # Check if there are at least four centroids to measure distances
        if len(centroids) >= 4:
            pairs = [
                (0, 2, 'Distance from Object 1 to Object 3', 'r'),  # Object 1 to Object 3
                (0, 3, 'Distance from Object 1 to Object 4', 'g'),  # Object 1 to Object 4
                (1, 2, 'Distance from Object 2 to Object 3', 'b'),  # Object 2 to Object 3
                (1, 3, 'Distance from Object 2 to Object 4', 'y'),  # Object 2 to Object 4
                (2, 3, 'Distance from Object 3 to Object 4', 'm'),  # Object 3 to Object 4
            ]

            for index1, index2, title, color in pairs:
                # Calculate Euclidean distance in pixels
                distance_px = distance.euclidean(centroids[index1], centroids[index2])
                print(f'Distance between Object {index1 + 1} and Object {index2 + 1}: {distance_px:.2f} pixels')

                # Create a new figure for each pair of objects
                plt.figure()
                plt.imshow(image, cmap='gray')
                plt.title(title)
                plt.axis('off')

                # Draw a line between the two centroids
                plt.plot([centroids[index1][0], centroids[index2][0]], 
                         [centroids[index1][1], centroids[index2][1]], color=color, linewidth=2)

                # Annotate the distance
                plt.text((centroids[index1][0] + centroids[index2][0]) / 2, 
                         (centroids[index1][1] + centroids[index2][1]) / 2, 
                         f'{distance_px:.2f} px', color=color, fontsize=12, fontweight='bold')

                # Plot centroids
                plt.plot(centroids[index1][0], centroids[index1][1], 'r*')  # Mark centroid of object 1
                plt.text(centroids[index1][0] + 5, centroids[index1][1], f'Obj {index1 + 1}', color='yellow', fontsize=12, fontweight='bold')
                plt.plot(centroids[index2][0], centroids[index2][1], 'r*')  # Mark centroid of object 2
                plt.text(centroids[index2][0] + 5, centroids[index2][1], f'Obj {index2 + 1}', color='yellow', fontsize=12, fontweight='bold')

                plt.show()

        else:
            print('Not enough objects detected to measure distances.')

    except Exception as e:
        print(f'Error: {e}')

# DICOM file name
dicom_filename = 'bar.dcm'  # Replace with the path to your DICOM file
process_dicom_image(dicom_filename)
