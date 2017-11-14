import cv2
import numpy as np

def detect(image, lower, upper):
    # find the colors within the specified boundaries and apply
    # the mask, in range --> white, out of range --> black
    mask = cv2.inRange(image, lower, upper)

    # calculate edges on the mask
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    return mask, edges


range_val = 50

# BGR
green = [49, 130, 57]
red = [66, 59, 196], # good
yellow = [100, 199, 203]
blue = [128, 101, 10], # pretty good
purple = [62, 31, 98] # terrible

color = np.array(red)
lower = color - np.array([1, 1, 1]) * range_val
upper = color + np.array([1, 1, 1]) * range_val

# using webcam, not raspberry pi camera
cap = cv2.VideoCapture(1)
cap.set(3, 360)
cap.set(4, 240)

while True:
	ret, image = cap.read()
	mask, edges = detect(image, lower, upper)
	kernel = np.ones((3, 3), np.uint8)
	erosion = cv2.erode(mask, kernel, iterations=1)

	
	_, contours, hierarchy = cv2.findContours(mask, 1, 2)
	if contours:
		for cnt in contours:
			#cnt = contours[0]
			M = cv2.moments(cnt)
			if M['m00'] != 0:
				cx = int(M['m10'] / M['m00'])
				cy = int(M['m01'] / M['m00'])
				print('x:', x, ' y:', y)

				cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

	cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

	cv2.imshow("Image", image)
	cv2.imshow("Mask", erosion)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()