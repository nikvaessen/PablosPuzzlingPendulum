from __future__ import division

import cv2
import time
import picamera
import numpy as np
import time


class Camera:
	
	def __init__(self, resolution):
		self.camera = picamera.PiCamera()
		self.resolution = resolution
		self.camera.resolution = resolution
		#self.camera.framerate = 24
		
	def capture(self):
		image = np.empty((240 * 320 * 3,), dtype=np.uint8)
		self.camera.capture(image, 'bgr')
		image = image.reshape((240, 320, 3))
		return image


class Detector:
	
	def __init__(self, lower, upper):
		self.lower = lower
		self.upper = upper
		
	def detect(self, image):
		# find the colors within the specified boundaries and apply
		# the mask, in range --> white, out of range --> black
		mask = cv2.inRange(image, lower, upper)
		
		# calculate edges on the mask
		edges = cv2.Canny(mask, 50, 150, apertureSize=3)

		# calculate lines on the mask image
		lines = cv2.HoughLines(edges, 1, np.pi/360, 50)
		
		# calculate most likely degree based on the radian
		# angle returned from the lines
		if lines is None:
			raise ValueError("No lines detected")
		
		return sum([x[1] for x in lines]) / len(lines)


if __name__ == "__main__":
	from datetime import datetime
	
	debug = True
	
	lower = np.array([110, 135, 200])
	upper = np.array([140, 165, 230])
	resolution = (320, 240)
	
	det = Detector(lower, upper)
	cam = Camera(resolution)
	
	while True:
		time.sleep(1)
		try:
			image = cam.capture()
			if debug:
				t = datetime.time(datetime.now())
				print(t)
				cv2.imwrite("debug/img_" + str(t) +".jpg", image)
			angle = det.detect(cam.capture())
			print(angle)
		except ValueError:
			print("No angle measured")
