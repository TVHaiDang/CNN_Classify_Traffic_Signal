import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
import json

class Data:

    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None
        self.X_test = None
        self.Y_test = None
    

    def read_raw_data(self, path):
        with open(path, mode = "rb") as f:
            data = pickle.load(f)
        return data

    
    def get_data(self):
        train_data = self.read_raw_data("dataset/train.p")
        val_data = self.read_raw_data("dataset/valid.p")
        test_data = self.read_raw_data("dataset/test.p")

        self.X_train = train_data["features"]
        self.Y_train = train_data["labels"]
        self.X_val = val_data["features"]
        self.Y_val = val_data["labels"]
        self.X_test = test_data["features"]
        self.Y_test = test_data["labels"]


    def shuffle(self):
        self.X_train, self.Y_train = shuffle(self.X_train, self.Y_train)


    def read_json_file(self, path):
        with open(path) as f:
            data = json.load(f)
        return data
    
    
    def normalize_data(self):
        self.X_train = self.X_train.astype("float")/255.0
        self.X_val = self.X_val.astype("float")/255.0
        self.X_test = self.X_test.astype("float")/255.0
    

    def label_data(self):
        lb = LabelBinarizer()
        self.Y_train = lb.fit_transform(self.Y_train)
        self.Y_val = lb.fit_transform(self.Y_val)


