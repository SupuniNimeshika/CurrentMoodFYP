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
    final_result = {}
    j=0
    prediction_result = np.divide(final_prediction,division)
    for result in prediction_result:
        final_result.update({mood[j]:result})
        j=j+1
    return prediction_result

