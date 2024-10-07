# from model import *
from preprocess import *
from keras.models import load_model
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def predict(model, input):
    input = np.array(input)
    if input.ndim == 1:
        input = input.reshape(1, -1)
    predictions = model.predict(input)
    i = np.argmax(predictions, axis=1)
    
    return predictions, class_map[i[0]]

class_map = {}
map = {'BENIGN': 0, 'DDoS': 1, 'Web Attack � Brute Force': 2, 'Web Attack � Sql Injection': 3, 'Web Attack � XSS': 4}
for i,j in map.items():
    class_map[j] = i

# model = load_model('pretrained_model/ptm.h5')
# input = x_tst.iloc[0]
# print(input, y_tst)
# output_prob, output_class = predict(model,input)
# print(output_prob, output_class)
