from flask import Flask, request, jsonify
import json
import requests
import ocr as ocr
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle
import unix_time as ut
from text_processing import pre_process
import emoticon as emo
import numpy as np
import weighted_algo as wa

app = Flask(__name__)
model=load_model("model/CNN-model.hdf5")
graph = tf.get_default_graph()

def prediction_model(text):
    MAX_LENGTH =200
    with open("model.pickle","rb") as f:
        tokenizer = pickle.load(f)

    #happy, sad, calm, angry
    text2='i am very happy'
    pre_process_text = pre_process(text)
    sequences = tokenizer.texts_to_sequences([pre_process_text])
    data = pad_sequences(sequences,maxlen=60,padding='post')
    print(data)


    with graph.as_default():
        predictionCNN =model.predict(data)
        print('--------------CNN------------------')
        return np.squeeze(predictionCNN,axis=0)

@app.route('/')
def hello_world():
    return 'supuni'


@app.route('/accessToken/<token>' , methods=['POST'])
def fb_data(token):
    print ('access')
    print(str(token))
    unixtime=ut.get_24_hours_time()
    re = 'https://graph.facebook.com/v3.2/me?fields=posts.since('+unixtime+')%7Bmessage%2Cfull_picture%7D&access_token='+token
    me1 = requests.get(re)
    # f1 =requests.get(friends)
    predictions = []
    data = me1.json()
    json_array = json.dumps(data)
    a = json.loads(json_array)
    mood_result=[]
    print(a)
    try:
        b=a['posts']
        data=b['data']
        predictions = iterate_post_array(data)
        mood_result=wa.weighted_post(predictions)
    except KeyError:
        print('---------------------There are no any post within 24 hours----------------------------')

# For User Profiling Part
    re1 = 'https://graph.facebook.com/v3.2/me?fields=likes%7Bartists_we_like%7D&access_token='+token
    me2 = requests.get(re1)
    dataInfo = me2.json()
    json_array_info = json.dumps(dataInfo)
    info = json.loads(json_array_info)
    print(info)
    user= {'user profile':info,'mood':mood_result}
    return jsonify(user)

def iterate_post_array(data):
    mood_prediction =[]

    for post in data:
        image_prediction = None
        emoticon_prediction=None
        message_prediction=None
        try:
            image_url = post['full_picture']
            image_post = ocr.url_to_image(image_url)
            text = ocr.image_to_text(image_post)
            image_prediction=prediction_model(text)
            print('INFO---getfull picture')
        except KeyError:
            print('Error picture')
            image_prediction = None

        try:
            post_message = post['message']
            emoti = emo.extract_emojis(post_message)
            if(emoti is not ''):
                emoticon_prediction=emo.emoticon_result_calculation(emoti)
                print('INFO---getfull emoticons')
            else:
                emoticon_prediction = None
            message_prediction = prediction_model(post_message)
            print('INFO---getfull message')

        except KeyError:
            message_prediction = None
        print(message_prediction)
        print(emoticon_prediction)
        print(image_prediction)

        if(message_prediction is None and image_prediction is not None):
            text_prediction = image_prediction
        elif(message_prediction is not None and image_prediction is None):
            text_prediction = message_prediction
        else:
            text_prediction=np.multiply(0.5,message_prediction)+np.multiply(0.5,image_prediction)


        if(emoticon_prediction is not None):
            total_prediction=np.multiply(0.4,text_prediction)+np.multiply(0.6,emoticon_prediction)
        else:
            total_prediction=text_prediction

        print('text-prediction'+str(text_prediction))
        print('total-prediction'+str(total_prediction))

        mood_prediction.append(total_prediction.tolist())
        print('mood-prediction'+str(mood_prediction))

    return mood_prediction


if __name__ == '__main__':
    app.run(host= '0.0.0.0' , port=9091, debug=True)

