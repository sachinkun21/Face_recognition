import pickle
import numpy as np
from src.embeddings import create_embeddings,get_embedding
from src.extract_faces import crop_faces


def predict_ml(image_path, input_name, model_path, le_path):
    face_array = crop_faces(image_path)
    face_embedding = get_embedding(face_array)
    face_embedding = np.expand_dims(face_embedding, axis=0)

    # loading Label
    loaded_model = pickle.load(open(model_path, 'rb'))
    prediction = loaded_model.predict(face_embedding)
    probability= loaded_model.predict_proba(face_embedding)[0]

    # loading label-Enocoder
    labelenocoder = pickle.load(open(le_path, 'rb'))
    prediction = labelenocoder.inverse_transform(prediction)


    print("Predicted {} , Actual {} with probability of {}%".format(prediction, input_name, max(probability)*100))

predict_ml('dataset/train/jerry_seinfeld/httpgraphicsnytimescomimagessectionmoviesfilmographyWireImagejpg.jpg', "jerry","models/RandomForest.sav", "models/LabelEncoder.sav")