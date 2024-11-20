import cv2
import numpy as np

threshold = 0.2
kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
x_co = 0
y_co = 0
hsv = None
H = 0
S = 0
V = 0
thr_H = 127 * threshold
thr_S = 255 * threshold
thr_V = 255 * threshold

# Mouse callback function to capture pixel HSV values on click
def on_mouse(event, x, y, flags, param):
    global x_co, y_co, H, S, V, hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        x_co = x
        y_co = y
        p_sel = hsv[y_co][x_co]
        H = p_sel[0]
        S = p_sel[1]
        V = p_sel[2]

# Load the image
src = cv2.imread("C:/Users/KLENVEN/Documents/pengolahan cinta/46451a05230014c05acd1c4005cf7f67.jpg")  # Change to the path of your image
src = cv2.blur(src, (3, 3))  # Apply blur
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)  # Convert the image to HSV color space

# Create windows
cv2.namedWindow("original", 1)
cv2.namedWindow("mask", 2)
cv2.setMouseCallback("original", on_mouse, 0)

while True:
    # Set the color range for masking based on the selected pixel
    min_color = np.array([H - thr_H, S - thr_S, V - thr_V])
    max_color = np.array([H + thr_H, S + thr_S, V + thr_V])
    
    # Create a mask that highlights the selected color range
    mask = cv2.inRange(hsv, min_color, max_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)
    
    # Display HSV values on the mask
    cv2.putText(mask, "H:" + str(H) + " S:" + str(S) + " V:" + str(V), 
                (10, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), thickness=1)
    
    # Display the original image and the mask
    cv2.imshow("original", src)
    cv2.imshow("mask", mask)
    
    # Segmented output (optionally show)
    src_segmented = cv2.add(src, src, mask=mask)
    # cv2.imshow("segmented", src_segmented)
    
    # Break the loop when 'Esc' key is pressed
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()