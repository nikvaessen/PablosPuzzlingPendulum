import cv2
import numpy as np

image = cv2.imread('test_images/bottomPendu.JPG')
image = cv2.resize(image, (1200, 600))

# create NumPy arrays from the boundaries
# order: blue, green, red
lower = np.array([110, 135, 200])
upper = np.array([140, 165, 230])

# find the colors within the specified boundaries and apply
# the mask, in range --> white, out of range --> black
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)

# save images
cv2.imwrite("out1.png", image)
cv2.imwrite("out2.png", output)

# calculate edges
edges = cv2.Canny(mask, 50, 150, apertureSize=3)
cv2.imwrite("out3.png", edges)

# calculate lines and display them on original image
lines = cv2.HoughLines(edges, 1, np.pi/360, 50)

if lines is None:
    print("No lines :(")
    exit()

for i in range(0, lines.shape[0]):
    for rho, theta in lines[i]:
        print("Degree: " + str(theta * 180 / np.pi))

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000 * -b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * -b)
        y2 = int(y0 - 1000 * a)

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)


cv2.imwrite('out4.png', image)
