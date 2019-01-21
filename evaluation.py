import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score

models =[
    LogisticRegressionCV(),
    LinearSVC(),
    MultinomialNB(),
    RandomForestClassifier(n_estimators=200,max_depth=3,random_state=0),
    BernoulliNB(),
    RidgeClassifier(),
    AdaBoostClassifier(),
    Perceptron(),
    PassiveAggressiveClassifier(),
]

CV =5
cv_df =pd.DataFrame(index=range(CV*len(models)))
entries =[]
for model in models:
    model_name =model.__class__.__name__
    accuracies =cross_val_score(model,X_train_tfidf,y_train,scoring='accuracy',cv=CV)
    for fold_idx, accuracy in enumerate(accuracy):
        entries.append((model_name,fold_idx,accuracy))
    cv_df = pd.DataFrame(entries,columns=['model_name','fold_idx','accuracy'])