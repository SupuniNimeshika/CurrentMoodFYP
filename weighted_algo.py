import numpy as np

def weighted_post(predction_list):
    mood=['happy','sad','calm','angry']
    post_count = len(predction_list)
    print(post_count)
    division = sum(range(1,post_count+1))
    print(division)
    i =0
    final_prediction =[0,0,0,0]
    for prediction in predction_list:
        multiply = np.multiply(post_count-i,prediction)
        print(multiply)
        final_prediction = np.array(final_prediction)+np.array(multiply)
        i=i+1
    print(final_prediction)
    prediction_result = np.divide(final_prediction,division)
    return prediction_result

def append_mood(final_result):
    final_label_result ={}
    mood = ['happy', 'sad', 'calm', 'angry']
    j = 0
    for result in final_result:
        print('--------------------------------------------------------')
        print(result)

        final_label_result.update({mood[j]:result})
        j=j+1
    return final_label_result


