import cv2
import numpy as np

from mtcnn.mtcnn import MTCNN
from src.embeddings import get_embeddings

import pickle

# create the detector, using default weights
detector = MTCNN()

cap = cv2.VideoCapture(0)

def predict_ml(face_array, input_name, model_path, le_path):
    face_embedding = get_embeddings(face_array)

    face_embedding = np.expand_dims(face_embedding, axis=0)

    # loading Label
    loaded_model = pickle.load(open(model_path, 'rb'))
    prediction = loaded_model.predict(face_embedding)
    probability= loaded_model.predict_proba(face_embedding)[0]

    # loading label-Enocoder
    labelenocoder = pickle.load(open(le_path, 'rb'))
    prediction = labelenocoder.inverse_transform(prediction)


    print("Predicted {} , Actual {} with probability of {}%".format(prediction, input_name, max(probability)*100))
    return prediction,max(probability)*100


def detect_face(image_array):
    # detect faces in the image
    results = detector.detect_faces(image_array)
    list_of_faces = []

    for res in results:
        # extract the bounding box from the first face
        x1, y1, width, height = res['box']

        # convert the co-ordinates into cropping format
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # crop and extract the face
        face = image_array[y1:y2, x1:x2]

        list_of_faces.append((face,(x1,y1,x2,y2)))
    return list_of_faces


prev_embed = None
frame_count = 0
while True:
    ret, frame = cap.read()

    frame_count += 1
    # Get all faces on current frame
    faces = detect_face(frame)

    if prev_embed is not None:

     for face in faces:
        if len(face) > 0:

            crop_face = face[0]
            face_co_ords = face[1]
            if prev_embed is not None:

                cv2.rectangle(frame,face_co_ords[:2], face_co_ords[2:], (255, 0, 0), 2)

                pred, prob = predict_ml(crop_face, "ragnar", "models/RandomForest.sav", "models/LabelEncoder.sav")
                label = pred[0]+" "+ str(prob)

            cv2.putText(frame, label,  (face_co_ords[0],face_co_ords[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)



    cv2.imshow("Face detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()