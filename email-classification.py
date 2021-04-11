#!/usr/bin/env python
# coding: utf-8

# In[37]:


# ## import all dependencies

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
import pickle 

# ## read the data and replace null values with a null string

# In[11]:


df1 = pd.read_csv("spamham.csv")
df = df1.where((pd.notnull(df1)), '')


# ## Categorize Spam as 0 and Not spam as 1 

# In[12]:


df.loc[df["Category"] == 'ham', "Category",] = 1
df.loc[df["Category"] == 'spam', "Category",] = 0


# ## split data as label and text . System should be capable of predicting the label based on the  text

# In[13]:


df_x = df['Message']
df_y = df['Category']


# ## split the table - 80 percent for training and 20 percent for test size

# In[14]:


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.8, test_size=0.2, random_state=4)


# ## feature extraction, coversion to lower case and removal of stop words using TFIDF VECTORIZER

# In[15]:


tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_trainFeat = tfvec.fit_transform(x_train)
x_testFeat = tfvec.transform(x_test)


# ## SVM is used to model

# In[16]:


y_trainSvm = y_train.astype('int')
classifierModel = LinearSVC()
classifierModel.fit(x_trainFeat, y_trainSvm)
predResult = classifierModel.predict(x_testFeat)
print("x_trainFeat", x_trainFeat)
print("predResult \n ", predResult)
message_transform = tfvec.transform(["Ok lar... Joking wif u oni..."])
print("message_tranform", message_transform)
message_predict = classifierModel.predict(message_transform)
if message_predict == 0:
    print("spam")
else:
    print("ham")
# ## GNB is used to model

# In[17]:


y_trainGnb = y_train.astype('int')
classifierModel2 = MultinomialNB()
classifierModel2.fit(x_trainFeat, y_trainGnb)
predResult2 = classifierModel2.predict(x_testFeat)


# ## Calc accuracy,converting to int - solves - cant handle mix of unknown and binary

# In[21]:


y_test = y_test.astype('int')
actual_Y = y_test.to_numpy()


# ## Accuracy score using SVM

# In[24]:


print("~~~~~~~~~~SVM RESULTS~~~~~~~~~~")
print("Accuracy Score using SVM: {0:.4f}".format(accuracy_score(actual_Y, predResult)*100))


# ## FScore MACRO using SVM

# In[25]:


print("F Score using SVM: {0: .4f}".format(f1_score(actual_Y, predResult, average='macro')*100))
cmSVM=confusion_matrix(actual_Y, predResult)


# ## "[True negative  False Positive\nFalse Negative True Positive]"

# In[26]:


print("Confusion matrix using SVM:")
print(cmSVM)


# ## Accuracy score using MNB

# In[27]:


print("~~~~~~~~~~MNB RESULTS~~~~~~~~~~")
print("Accuracy Score using MNB: {0:.4f}".format(accuracy_score(actual_Y, predResult2)*100))


# ## FScore MACRO using MNB

# In[28]:


print("F Score using MNB:{0: .4f}".format(f1_score(actual_Y, predResult2, average='macro')*100))


# In[29]:


cmMNb=confusion_matrix(actual_Y, predResult2)


# ## "[True negative  False Positive\nFalse Negative True Positive]"

# In[30]:


print("Confusion matrix using MNB:")
print(cmMNb)


# In[39]:


project_name='email-classification'


# In[ ]:

