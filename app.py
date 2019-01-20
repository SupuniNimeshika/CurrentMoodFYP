from flask import Flask, request, jsonify
import json
import requests
import ocr as ocr


app = Flask(__name__)


@app.route('/')
def hello_world():
    token='EAAbJHUYwk6YBAL0ZBljr7rXw4X6dycX5iL5e1hS0hhIVv9XIsFvuZCENsVOwPphxDzwqZBZCv3knKv7CF7edza52ZAFmPZCJ0Gn7XBGZBiuD5ENTTsU4JSRYDEuz5opnPp5SlwRgpKVIXlERYAr53HrZBHGHwSqeZASa013M43jSAzHwqOd4PY7owlaZBZAm0ZCZB08aQyXI3F3l8KfduDnFH3mU4'
    # token='EAAbJHUYwk6YBAPDiXQkGWvGyr9wBNVu4T5JCHUhDqCg8DMxGBw1pixjLVTC9EW9EEoUbzS6DzZBvihXJZAIegVaBpT1OnLtZCZBc1Gfc2Fii83QdLcybIMCinQ2AZBn8i7cyp7NrrVhLqkZBPVuhEAVWDikDQ4jwUYtMcRnxfNbUzYame394RqqwZBlrBx0rkzys7zeB1FO1vfW5FG0etFDiBojYyoVs4cZD'
    print ('Supuni')
    re = 'https://graph.facebook.com/v3.2/me/posts?fields=comments%2Cmessage%2Cfull_picture&access_token=' + token
    me1 = requests.get(re)
    # f1 =requests.get(friends)

    print(me1.json())
    data =me1.json()
    json_array =json.dumps(data)
    a= json.loads(json_array)
    store_list = []
    print(a['data'])

    for item in a['data']:


        try:
            image_url = item['full_picture']
            print(image_url)
        except KeyError:
            print ("Haven't full picture")



    return jsonify(me1.text)



@app.route('/accessToken/<token>' , methods=['POST'])
def fb_data(token):
    print ('access')
    print(str(token))
    re = 'https://graph.facebook.com/v3.2/me/posts?fields=comments%2Cmessage%2Cfull_picture&access_token='+token
    me1 = requests.get(re)
    # f1 =requests.get(friends)

    data = me1.json();
    json_array = json.dumps(data);
    a = json.loads(json_array)
    store_list = []
    print(a['data'])

    for item in a['data']:

        try:
            image_url = item['full_picture']
            store_list.append(image_url)
        except KeyError:
            print("Haven't full picture")

    image =ocr.url_to_image(store_list[5])
    text=ocr.image_to_text(image)
    mc.data_predict(text)
    print(text)

    return jsonify(store_list)



if __name__ == '__main__':
    app.run(host= '0.0.0.0' , port=9091, debug=True)
