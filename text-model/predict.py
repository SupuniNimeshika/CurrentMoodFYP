from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import data_extract as data_source
import pickle
from ploting_graph import plot_history

modelNN = load_model("NN-model.hdf5")
modelCNN = load_model("CNN-model.hdf5")

MAX_LENGTH =200
with open("model.pickle","rb") as f:
    tokenizer = pickle.load(f)

# Testing with new data
text=['Never put your happiness in someone else’s hands.','happy']

sequences = tokenizer.texts_to_sequences(text)
data = pad_sequences(sequences,maxlen=100,padding='post')
print(data)
predictionNN =modelNN.predict(data)
print('--------------NN------------------')
print(predictionNN)
predictionCNN =modelCNN.predict(data)
print('--------------CNN------------------')
print(predictionCNN)
