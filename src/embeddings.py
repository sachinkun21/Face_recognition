from keras.models import load_model
import numpy as np
from src.load_data import load_array


# loading the facenet model
emb_model = load_model('models/facenet_keras.h5')

# summarize input and output shape
print(emb_model.inputs)
print(emb_model.outputs)


def get_embedding(face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)

    y_emb = emb_model.predict(samples)

    # get embedding
    embedding = y_emb[0]
    return embedding


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
