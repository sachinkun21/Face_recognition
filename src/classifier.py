# for deep learning models
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import  Adam

import pickle

# for machine learning models
from sklearn.svm import  SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# for processing y labels
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical
from src.load_data import load_array
import  numpy as np


class NeuralNetClassifier:
    def __init__(self,batch_size = 8, epochs = 5):
        self.input_shape = None
        self.num_classes = None
        self.model = None
        self.train_X, self.train_y, self.test_X, self.test_y = None, None, None, None

        self.batch_size = batch_size
        self.epochs = epochs

        self.le = LabelEncoder()
        self.oe = OneHotEncoder()

    def build(self):
        model = Sequential()
        model.add(Dense(1024, activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        self.model = model

    def process_y(self,y):
        y = self.le.fit_transform(y)
        y = y.reshape(-1,1)
        y = self.oe.fit_transform(y).toarray()
        return y

    def process_data(self):
        self.train_y = self.process_y(self.train_y)

        self.test_y = self.process_y(self.test_y)

        self.input_shape = self.train_X.shape
        self.num_classes = self.train_y.shape[1]
        print(self.test_X.shape, self.test_y.shape)

        self.build()

    def fit_model(self,embedding_path ):
        self.train_X, self.train_y, self.test_X, self.test_y = load_array(embedding_path)
        self.process_data()

        history = self.model.fit( self.train_X, self.train_y, batch_size= self.batch_size, epochs= self.epochs,
                              verbose=1, validation_data=(self.test_X, self.test_y ))

        print(history.history['accuracy'])

    def save_model(self, path_to_save='../models/'):
        self.model.save(path_to_save+"nn_model.h5")
        print("Saved model to disk")

    def predict_face(self, embedding):
        self.model.predict(embedding)




class MLClassifier:
    def __init__(self,classifier_type):
        self.model_type = classifier_type
        self.model = None
        self.train_X, self.train_y, self.test_X, self.test_y = None, None, None, None
        self.le = LabelEncoder()

    def process_data(self):
        self.train_y = self.le.fit_transform(self.train_y)
        self.test_y = self.le.fit_transform(self.test_y)

    def fit_model(self,embedding_path):
        self.train_X, self.train_y, self.test_X, self.test_y = load_array(embedding_path)
        self.process_data()

        if self.model_type == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators = 100)
        elif self.model_type == 'SupportVector':
            self.model = SVC(kernel='linear', probability=True)
        elif self.model_type == 'DecisionTree':
            self.model = DecisionTreeClassifier()
        else:
            return  "Incorrect ML Model type"

        print("-------Training {} classifier-------".format(self.model_type))
        self.model.fit(self.train_X,self.train_y)
        print("Training Accuracy", self.model.score(self.train_X,self.train_y))
        print("Validation Accuracy", self.model.score(self.test_X, self.test_y))
        print("---saving model in models directory---")
        filename = "models/{}.sav".format(self.model_type)
        pickle.dump(self.model, open(filename, 'wb'))
        pickle.dump(self.le, open('models/LabelEncoder.sav', 'wb'))


    def predict_face(self, embedding):
        embedding = np.expand_dims(embedding,axis=0)
        predicted = self.model.predict(embedding)
        predicted = self.le.inverse_transform(predicted)
        return predicted





#
# data_path = '../dataset/saved_arrays/embeddings-dataset.npz'
# NN1 = NeuralNetClassifier(data_path)
# NN1.fit_model(save_model=True)
#
# ml1 = MLClassifier("SupportVector", data_path)
# ml1.fit_model()