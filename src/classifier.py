# for deep learning models
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import  Adam

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


class NeuralNetClassifier:
    def __init__(self, embedding_path,batch_size = 8, epochs = 5):
        self.input_shape = None
        self.num_classes = None
        self.model = None
        self.train_X, self.train_y, self.test_X, self.test_y = load_array(embedding_path)

        self.batch_size = batch_size
        self.epochs = epochs

        self.le = LabelEncoder()
        self.oe = OneHotEncoder

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


    def process_data(self):
        self.train_y = self.le.fit_transform(self.train_y )
        self.train_y = self.oe.fit_transform(self.train_y.reshape(-1,1)).toarray()

        self.test_y = self.le.fit_transform(self.test_y )
        self.test_y = self.oe.fit_transform(self.test_y.reshape(-1,1)).toarray()

        self.input_shape = self.train_X.shape
        self.num_classes = self.train_y.shape[1]
        print(self.test_X.shape, self.test_y.shape)

        self.build()

    def fit_model(self):
        self.process_data()

        history = self.model.fit( self.train_X, self.train_y, batch_size= self.batch_size, epochs= self.epochs,
                              verbose=1, validation_data=(self.test_X, self.test_y ))
        print(history.history['accuracy'])

    def predict_model(self, image):
        pass



class MLClassifier:
    def __init__(self,classifier_type, embedding_path):
        self.svc= SVC(kernel='linear', probability=True)
        self.rfc = RandomForestClassifier(n_estimators = 100)
        self.dtc = DecisionTreeClassifier()
        self.model = classifier_type
        self.train_X, self.train_y, self.test_X, self.test_y = load_array(embedding_path)
        self.le = LabelEncoder()

    def process_data(self):
        self.train_y = self.le.fit_transform(self.train_y)
        self.test_y = self.le.fit_transform(self.test_y)

    def fit_model(self):
        self.process_data()
        print("-------Training {} classifier-------".format(self.model))
        if self.model=='RandomForest':
            self.rfc.fit(self.train_X,self.train_y)
            print("Training Accuracy", self.rfc.score(self.train_X,self.train_y))
            print("Validation Accuracy", self.rfc.score(self.test_X, self.test_y))

        elif self.model=='SupportVector':
            self.svc.fit(self.train_X,self.train_y)
            print("Training Accuracy", self.svc.score(self.train_X,self.train_y))
            print("Validation Accuracy", self.svc.score(self.test_X, self.test_y))

        elif self.model=='DecisionTree':
            self.dtc.fit(self.train_X,self.train_y)
            print("Training Accuracy", self.dtc.score(self.train_X,self.train_y))
            print("Validation Accuracy", self.dtc.score(self.test_X, self.test_y))





data_path = '../dataset/saved_arrays/embeddings-dataset.npz'
# NN1 = NeuralNetClassifier(data_path)
# NN1.fit_model()

ml1 = MLClassifier("DecisionTree", data_path)
ml1.fit_model()