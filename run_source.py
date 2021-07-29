from tensorflow.keras.models import load_model
from reprocessing_data import Data
from model import Model
import numpy as np
import matplotlib.pyplot as plt
class Run:

    def __init__(self):
        self.data = Data()
        self.model = None
        self.completed_model = None


    def run_model(self):
        model = Model()
        self.data.get_data()
        self.data.shuffle()
        self.data.normalize_data()
        self.data.label_data()
        self.model = model.run_model(self.data.X_train, self.data.Y_train, self.data.X_val, self.data.Y_val)


    def save_model(self):
        self.model.save("weights.h5")

    
    def load_model(self):
        self.completed_model = load_model("weights.h5")


    def predict_data(self):
        
        self.data.get_data()
        self.data.shuffle()
        self.data.normalize_data()
        self.data.label_data()
        self.load_model()
        class_name = self.data.read_json_file("config/labels.json")
        matrix_result = self.completed_model.predict(self.data.X_test[8:9])
        value_matrix_result = np.argmax(matrix_result)
        print("label: "+ str(self.data.Y_test[value_matrix_result]))
        print("category: " + class_name[str(value_matrix_result)])
        plt.imshow(self.data.X_test[8])
        plt.show()


if __name__=="__main__":
    x = Run()
    x.load_model()
    x.predict_data()





