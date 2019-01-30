import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle
from text_processing import pre_process

def prediction_model(model,text):
    # modelNN = load_model("model/NN-model.hdf5")
    modelCNN = model

    MAX_LENGTH =200
    with open("model.pickle","rb") as f:
        tokenizer = pickle.load(f)

    #happy, sad, calm, angry
    text2='i am very happy'
    pre_process_text = pre_process(text2)
    sequences = tokenizer.texts_to_sequences([pre_process_text])
    data = pad_sequences(sequences,maxlen=60,padding='post')
    print(data)
    # predictionNN =modelNN.predict(data)
    # print('--------------NN------------------')
    # print(predictionNN)

    with tf.Graph().as_default():
        predictionCNN =model.predict(data)
        print('--------------CNN------------------')
        print(predictionCNN)
