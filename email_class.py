#!/usr/bin/env python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
# from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

def message_classifier(message):
    """This function return which category message belongs to"""
    df1 = pd.read_csv("spamham.csv")
    df = df1.where((pd.notnull(df1)), '')

    df.loc[df["Category"] == 'ham', "Category",] = 1
    df.loc[df["Category"] == 'spam', "Category",] = 0


    df_x = df['Message']
    df_y = df['Category']

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

    tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    x_trainFeat = tfvec.fit_transform(x_train)
    x_testFeat = tfvec.transform(x_test)

    y_trainSvm = y_train.astype('int')
    classifierModel = LinearSVC()
    classifierModel.fit(x_trainFeat, y_trainSvm)
    predResult = classifierModel.predict(x_testFeat)

    # predicting actual mesage
    message_transform = tfvec.transform([message])
    message_predict = classifierModel.predict(message_transform)
    if message_predict == 0:
        return "Spam"
    else:
        return "Ham"

    print("~~~~~~~~~~SVM RESULTS~~~~~~~~~~")
    print("Accuracy Score using SVM: {0:.4f}".format(accuracy_score(actual_Y, predResult)*100))

    print("F Score using SVM: {0: .4f}".format(f1_score(actual_Y, predResult, average='macro')*100))


