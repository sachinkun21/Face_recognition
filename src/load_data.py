import os
import numpy as np

from src.extract_faces import crop_faces
from src.extract_faces import save_face

from src.embeddings import get_embeddings


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
def save_face_array(train_dir , test_dir = None, path_to_save = '../dataset/saved_arrays/faces-dataset.npz'):
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




def create_embeddings(input_array_path, output_array_path):
    train_X, train_y, test_X, test_y = load_array(path_to_npz=input_array_path)

    print("-------Creating Embeddings of shape-------")
    embedding_train = []
    embedding_test = []
    for cropped_face in train_X:
        embedding = get_embedding(cropped_face)
        embedding_train.append(embedding)

    for cropped_face in test_X:
        embedding = get_embedding(cropped_face)
        embedding_test.append(embedding)

    embedding_train = np.asarray(embedding_train)
    embedding_test = np.asarray(embedding_test)

    print(embedding_train.shape, embedding_test.shape)
    print("*********Successfully created. Saving Embedding to npz array*********")
    np.savez_compressed(output_array_path, embedding_train, train_y, embedding_test, test_y)


def add_embedding(new_train_images, new_test_images , saved_embedding_path):
    embed_train_X, embed_train_y, embed_test_X, embed_test_y = load_array(path_to_npz=saved_embedding_path)

    new_embedding_train = []
    new_embedding_test = []

    new_face_X, new_face_y = load_dataset(new_train_images)
    new_test_X, new_test_y = load_dataset(new_test_images)

    for cropped_face in new_face_X:
        embedding = get_embeddings(cropped_face)
        new_embedding_train.append(embedding)

    for cropped_face in new_test_X:
        embedding = get_embeddings(cropped_face)
        new_embedding_test.append(embedding)

    new_embedding_train = np.asarray(new_embedding_train)
    new_embedding_test = np.asarray(new_embedding_test)

    final_embedding_train = np.concatenate([embed_train_X, new_embedding_train])
    final_embedding_test = np.concatenate( [embed_test_X, new_embedding_test])

    # labels:
    final_train_y = np.concatenate([embed_train_y, new_face_y])
    final_test_y = np.concatenate([embed_test_y, new_test_y])

    # print((embed_train_X).shape, (embed_test_X).shape, (embed_train_y).shape, (embed_test_y).shape)
    # print((final_embedding_train).shape, (final_embedding_test).shape, (final_train_y).shape, (final_test_y).shape)

    print("*********Successfully created. Saving new + old Embedding to npz array*********")
    np.savez_compressed(saved_embedding_path, final_embedding_train, final_train_y, final_embedding_test, final_test_y)

    # fitting new model
    # ml_model = MLClassifier('RandomForest')
    #
    # # calling fit_model on new embedding
    # ml_model.fit_model(saved_embedding_path)


# add new embeddings function test
# embed_path = '/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/saved_arrays/embeddings-dataset.npz'
# train_dir= '/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/train'
# test_dir = '/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/val'
# add_embedding(train_dir,test_dir, embed_path)


# function 1
# faces = load_faces('/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/train/ben_afflek/')
# print(len(faces))
# save_face(faces[0],"face_0.jpg")

# function 2
#load_dataset('/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/train/')

# function 3
# train_dir= '/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/train'
# test_dir = '/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/val'
# save_array(train_dir, test_dir)


#load_array('../dataset/saved_arrays/faces-dataset.npz')