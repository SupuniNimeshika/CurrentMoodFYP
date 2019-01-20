import random

def get_data():
    angryfile = open('../data/angry.txt','r')
    angry = angryfile.readlines()
    sadfile = open('../data/sad.txt','r')
    sad = sadfile.readlines()
    happyfile = open('../data/happy.txt','r')
    happy = happyfile.readlines()
    calmfile = open('../data/calm.txt','r')
    calm = calmfile.readlines()

    sad_spot = 0
    happy_spot = 1
    angry_spot = 2
    calm_spot = 3

    data = []
    for line in angry:
        str=line.translate({ord('\n'): None})
        str= 'angry|'+str
        data.append(str)

    for line in sad:
        str=line.translate({ord('\n'): None})
        str= 'sad|'+str
        data.append(str)

    for line in calm:
        str=line.translate({ord('\n'): None})
        str= 'calm|'+str
        data.append(str)

    for line in happy:
        str=line.translate({ord('\n'): None})
        str= 'happy|'+str
        data.append(str)

    random.shuffle(data)
    x = []
    y = []

    for d in data:
        label, text = d.split('|')
        if label=='happy':
            spot=happy_spot
        elif label=='sad':
            spot=sad_spot
        elif label=='calm':
            spot=calm_spot
        elif label=='angry':
            spot=angry_spot
        y.append(spot)
        x.append(text)

    return (x,y)