from src.load_data import save_array,load_array
from src.embeddings import create_embeddings,get_embedding
from src.classifier import NeuralNetClassifier,MLClassifier
from src.extract_faces import crop_faces
import  numpy as np



train_path = '/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/train/'
test_path = '/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/val/'
input_path = 'dataset/saved_arrays/faces-dataset.npz'

#save_array(train_path,test_path,path_to_save = array_path)

embedding_path = 'dataset/saved_arrays/embeddings-dataset.npz'

#create_embeddings(input_path, embedding_path)




def train():
    pass


def add_new_person():
    pass



### Prediction Code
ml_model = MLClassifier('RandomForest')
ml_model.fit_model(embedding_path)

def predict_ml(image_path, input_name):
    face_array = crop_faces(image_path)
    face_embedding = get_embedding(face_array)
    prediction = ml_model.predict_face(face_embedding)
    print("Predicted {} , Actual {}".format(prediction,input_name))

predict_ml('dataset/train/mindy_kaling/httpimagesnymagcomimagesdailymindykalingxjpg.jpg', "mindy kaling")




# dl_model = NeuralNetClassifier(batch_size = 8, epochs = 5)
# dl_model.fit_model(embedding_path)