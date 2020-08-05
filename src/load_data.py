import os
import numpy as np

from src.extract_faces import crop_faces
from src.extract_faces import save_face


# load saved_arrays and extract faces for all saved_arrays in a directory
def load_faces(directory):

    faces = list()
    # enumerate files
    for filename in os.listdir(directory):
        # path of image
        path_image = os.path.join(directory, filename)

        # get face
        face = crop_faces(path_image)

        # append in faces list
        faces.append(face)
    return faces


# load a dataset that contains one subdir for each class that in turn contains saved_arrays
def load_dataset(directory):

    X, y = list(), list()

    # enumerate folders, on per class
    for subdir in os.listdir(directory):
        # path
        _path = os.path.join(directory ,subdir)

        # skip any files that might be in the dir
        if not os.path.isdir(_path):
            # skip
            continue

        # load all faces in the subdirectory
        faces = load_faces(_path)

        # create labels
        labels = [subdir for i in range(len(faces))]

        # summarize progress
        print('-->>loaded %d examples for class: %s' % (len(faces), subdir))

        # store
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


# Calls the load data-set on directories passed as train and test. Saves the output in npz format
def save_array(train_dir , test_dir = None, path_to_save = '../dataset/saved_arrays/faces-dataset.npz'):
    train_X, train_y = load_dataset(train_dir)

    if test_dir is not None:
        test_X, test_y = load_dataset(test_dir)
    else:
        test_X = 0
        test_y = 0

    # save arrays to one file in compressed format
    np.savez_compressed(path_to_save, train_X, train_y, test_X, test_y)


def load_array(path_to_npz):

    data = np.load(path_to_npz)
    train_X, train_y, test_X, test_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Loaded: {}, {} ,{} ,{}  from {} '.format(train_X.shape, train_y.shape, test_X.shape, test_y.shape, path_to_npz))
    return train_X, train_y, test_X, test_y





# function 1
#faces = load_faces('/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/train/ben_afflek/')
#print(len(faces))
#save_face(faces[0],"face_0.jpg")

# function 2
#load_dataset('/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/train/')

# function 3
# train_dir= '/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/train'
# test_dir = '/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/val'
# save_array(train_dir, test_dir)


#load_array('../dataset/saved_arrays/faces-dataset.npz')