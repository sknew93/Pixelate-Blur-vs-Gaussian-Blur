import cv2
import numpy as np 

def blur_sample(image, factor=4): # higher the factor less the blur
	(h,w) = image.shape[:2]
	kh = h//factor
	kw = w//factor
	#has to be odd
	if kh%2 ==0:
		kh -= 1
	if kw%2 ==0:
		kw -= 1

	return cv2.GaussianBlur(image, (kh, kw), 0)

def resize_sample(img, factor=2): # factor=2 halves the size(to ease the display), 1 is orignal size
	resized_img = cv2.resize(img,(int(img.shape[1]/factor), int(img.shape[0]/factor)))
	print(resized_img.shape)
	return resized_img


def blur_pixel(image, blocks=10): #blocks = no. of rows/ cols needs to divide the image
	(h,w) = image.shape[:2]
	step_x = np.linspace(0, w, blocks+1, dtype='int')
	step_y = np.linspace(0, h, blocks+1, dtype = np.int32)

	for y in range(len(step_y)-1):
		for x in range(len(step_x)-1):
			sx = step_x[x]
			sy = step_y[y]
			ex = step_x[x+1]
			ey = step_y[y+1]

			(B, G, R) = cv2.mean(image[sy:ey, sx:ex])[:3] # will calculate mean along 3 induvidal chanels
			cv2. rectangle(image, (sx, sy), (ex, ey), (B, G, R), -1)
	return image


def draw_blocks(image, size=3): #size =3 for 3x3
	(h, w) = image.shape[:2]
	coord_x = np.linspace(0, w, size+1, dtype=np.int32)
	coord_y = np.linspace(0, h, size+1, dtype=np.int32)
	print(coord_x, coord_y)
	for i in range(1,size):
		cv2.line(image, (coord_x[i], 0), (coord_x[i], h), (0,255,0), 2)
		cv2.line(image, (0, coord_x[i]), (w, coord_x[i]), (0,255,0), 2)

	return image


img = cv2.imread('data/persons_1.jpg')
print(img.shape)

#Face Detection

#1. Create Cascade Classifier,contains features of face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# glasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses')

#2. Search for row and column for the face
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(	gray_img,
										scaleFactor = 1.06,
										minNeighbors = 5)

cv2.putText(img, f"The Haarcascade Classifier detects \"{len(faces)}\" faces.", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3) #times of defaultsize, text thickness

print(faces)

print(type(faces))
#3. Display Image with face box
blur_pixels = img.copy()
blur_samples = img.copy()
blur_all = img.copy()

for x,y,w,h in faces:
	img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

	face = blur_samples[y:y+h, x:x+w]
	blur_face = blur_sample(face)
	blur_samples[y:y+h, x:x+w] = blur_face

	face = blur_pixels[y:y+h, x:x+w]
	blur_face = blur_pixel(face)
	blur_pixels[y:y+h, x:x+w] = blur_face

	face = blur_all[y:y+h, x:x+w]
	blur_all = blur_sample(blur_all,45)
	blur_all[y:y+h, x:x+w] = face

#resize
draw_blocks(img,4)

cv2.imshow("Face Track", resize_sample(img,4))
cv2.imshow('Simple Blur', resize_sample(blur_samples,4))
cv2.imshow('Pixelation Blur', resize_sample(blur_pixels,4))
# cv2.imshow('All Blur', resize_sample(blur_all,4))


cv2.waitKey(0)
