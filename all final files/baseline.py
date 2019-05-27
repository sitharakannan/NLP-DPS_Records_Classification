
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk, re, pprint
from nltk import word_tokenize


# In[2]:


df = pd.read_csv('final.csv')


# In[3]:


from sklearn.utils import shuffle
df = shuffle(df)
train, test = train_test_split(df, test_size=0.2)
y_true = test['label']


# In[4]:


B_1=[]
B_2=[] 
B_3=[] 
B_4=[] 
B_5=[]

for index,row in train.iterrows(): 
    temp = row['Summary']
    if isinstance(temp, basestring): 
        temp=temp.split()
        if(row['label'] == 1): 
            for i in temp: 
                B_1.append(i)

        if(row['label'] == 2):
            for i in temp:
                B_2.append(i)

        if(row['label'] == 3):
            for i in temp:
                B_3.append(i)

        if(row['label'] == 4):
            for i in temp:
                B_4.append(i)

        if(row['label'] == 5):
            for i in temp:
                B_5.append(i)


# In[5]:


y_pred = []
for index,row in test.iterrows(): 
    temp = row['Summary'] 
    count=[0,0,0,0,0]
   
    if isinstance(temp, basestring): 
        temp=temp.split() 
        for i in temp:
            if(i in B_1): 
                count[0]+=1
            if(i in B_2): 
                count[1]+=1
            if(i in B_3): 
                count[2]+=1
            if(i in B_4): 
                count[3]+=1
            if(i in B_5): 
                count[4]+=1
    y_pred.append(count.index(max(count))+1)
    
print accuracy_score(y_true, y_pred)


# In[6]:


print classification_report(y_true, y_pred)

