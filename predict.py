from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle


modelNN = load_model("model/NN-model.hdf5")
modelCNN = load_model("model/CNN-model.hdf5")

MAX_LENGTH =200
with open("model.pickle","rb") as f:
    tokenizer = pickle.load(f)

# Testing with new data
text=['Never put happiness someone elses hands','happy']

sequences = tokenizer.texts_to_sequences(text)
data = pad_sequences(sequences,maxlen=100,padding='post')
print(data)
predictionNN =modelNN.predict(data)
print('--------------NN------------------')
print(predictionNN)
predictionCNN =modelCNN.predict(data)
print('--------------CNN------------------')
print(predictionCNN)
