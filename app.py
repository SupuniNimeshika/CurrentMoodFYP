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



    predictions = []
    mood_result = []
    time_range=''
    # final 18 hours
    since = ut.get_hours_time(24)
    until=ut.get_hours_time(6)
    eighteen_hour_url = 'https://graph.facebook.com/v3.2/me?fields=posts.until('+until+').since(' + since + ')%7Bmessage%2Cfull_picture%7D&access_token=' + token
    eighteen_hour_req = requests.get(eighteen_hour_url)
    eighteen_hour_data = eighteen_hour_req.json()
    eighteen_hour_json_array = json.dumps(eighteen_hour_data)
    eighteen_hour_arr = json.loads(eighteen_hour_json_array)
    try:
        eighteen_hour_post = eighteen_hour_arr['posts']
        eighteen_hour_data = eighteen_hour_post['data']
        eighteen_hour_predictions = iterate_post_array(eighteen_hour_data)
        eighteen_hour_mood_result = wa.weighted_post(eighteen_hour_predictions)
        time_range='to nearest 24'
    except KeyError:
        eighteen_hour_mood_result = [0, 0, 0, 0]
        print('---------------------There are no any post within 6th hour to 24th hour----------------------------')

    # next five hours
    since = ut.get_hours_time(6)
    until = ut.get_hours_time(1)
    five_hour_url = 'https://graph.facebook.com/v3.2/me?fields=posts.until('+until+').since(' + since + ')%7Bmessage%2Cfull_picture%7D&access_token=' + token
    five_hour_req = requests.get(five_hour_url)
    five_hour_data = five_hour_req.json()
    five_hour_json_array = json.dumps(five_hour_data)
    five_hour_arr = json.loads(five_hour_json_array)
    try:
        five_hour_post = five_hour_arr['posts']
        five_hour_data = five_hour_post['data']
        five_hour_predictions = iterate_post_array(five_hour_data)
        five_hour_mood_result = wa.weighted_post(five_hour_predictions)
        time_range = 'to nearest 6'
    except KeyError:
        five_hour_mood_result = [0, 0, 0, 0]
        print('---------------------There are no any post within 2nd hour to 6th hour----------------------------')

    # last hour
    since = ut.get_hours_time(1)
    last_hour_url = 'https://graph.facebook.com/v3.2/me?fields=posts.since(' + since + ')%7Bmessage%2Cfull_picture%7D&access_token=' + token
    last_hour_req = requests.get(last_hour_url)
    last_hour_data = last_hour_req.json()
    last_hour_json_array = json.dumps(last_hour_data)
    last_hour_arr = json.loads(last_hour_json_array)
    try:
        last_hour_post=last_hour_arr['posts']
        last_hour_data=last_hour_post['data']
        last_hour_predictions = iterate_post_array(last_hour_data)
        last_hour_mood_result=wa.weighted_post(last_hour_predictions)
        time_range='to last hour'
    except KeyError:
        last_hour_mood_result=[0,0,0,0]
        print('---------------------There are no any post within last hours----------------------------')



    print(last_hour_mood_result)
    print(five_hour_mood_result)
    print(eighteen_hour_mood_result)
    mood_result=np.multiply(0.7,last_hour_mood_result)+np.multiply(0.2,five_hour_mood_result)+np.multiply(0.1,eighteen_hour_mood_result)
    sum_result=sum(mood_result)
    mood_result_final=np.true_divide(mood_result,sum_result)
    mood_result_label=wa.append_mood(mood_result_final.tolist())

# For User Profiling Part
    re1 = 'https://graph.facebook.com/v3.2/me?fields=likes%7Bartists_we_like%7D&access_token='+token
    me2 = requests.get(re1)
    dataInfo = me2.json()
    json_array_info = json.dumps(dataInfo)
    info = json.loads(json_array_info)
    print(info)
    #append user info, mood, time range
    user= {'user profile':info,'mood':mood_result_label,'time':time_range}
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

