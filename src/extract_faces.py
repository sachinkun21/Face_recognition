
# function for face detection with mtcnn
from PIL import Image
import numpy as np
import cv2

from mtcnn.mtcnn import MTCNN


# create the detector, using default weights
detector = MTCNN()


def detect_face(image_array, num_of_faces = 1):

	# detect faces in the image
	results = detector.detect_faces(image_array)
	list_of_faces = []

	for i in results[:num_of_faces]:
		# extract the bounding box from the first face
		x1, y1, width, height = results[0]['box']

		# convert the co-ordinates into cropping format
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height

		# crop and extract the face
		face = image_array[y1:y2, x1:x2]

		list_of_faces.append(face)

	return list_of_faces


# extract a single face from a given photograph
def crop_faces(file, required_size=(160, 160), num_of_faces = 1):

	# load image from file
	image = Image.open(file)

	# convert to RGB, if needed
	image = image.convert('RGB')

	# convert to array
	image_array = np.asarray(image)

	# faces extracted from MTCNN
	list_of_faces = detect_face(image_array, num_of_faces)

	# new list to store resized faces
	list_of_resized_faces = []

	for face in list_of_faces:
		# resize array to the model size by reading the array into Pil image object
		image = Image.fromarray(face)
		image = image.resize(required_size)

		face_array = np.asarray(image)

		# add face array to new list
		list_of_resized_faces.append(face_array)

	return list_of_resized_faces[0] if num_of_faces == 1 else list_of_resized_faces


def save_face(array, name = "Cropped Face.jpg"):
	cv2.imwrite(name, cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


def plot_face(array):
	pass


# load the photo and extract the face
# pixels = crop_faces('../vikings-ragnar1200.jpg')

# save_face(pixels)
# plot_face(pixels)