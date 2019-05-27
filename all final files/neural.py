
# coding: utf-8

# In[1]:


import random
import string
import pandas as pd
import numpy as np


# In[2]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics


# In[3]:


import nltk


# In[4]:


import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# In[5]:


df = pd.read_csv('final.csv')


# In[6]:


df_1K = df


# In[7]:


print df_1K['label'].value_counts()


# ## Pre-processing
# Integer of frequency counts

# #### Count Dict

# In[8]:


word_count_dict = {}


# In[9]:


for index,row in df_1K.iterrows():
    s = row['Summary']
    words = nltk.word_tokenize(s)
    words=[word.lower() for word in words if word.isalpha()]
    for i in words:
        try:
            word_count_dict[i]+=1
        except:
            word_count_dict[i]=1


# In[10]:


wc_list = sorted(word_count_dict.items(), key=lambda x: x[1],reverse=True)


# In[11]:


wc_dict = {}
ct_no = 1
for item in wc_list:
    wc_dict[item[0]] = ct_no
    ct_no+=1


# #### Changing the input to freq-count form

# In[14]:


X_list = []
y_list = []
for index,row in df_1K.iterrows():
    y_list.append(int(row['label']))
    iwords = []
    s = row['Summary']
    words = nltk.word_tokenize(s)
    words=[word.lower() for word in words if word.isalpha()]
    for item in words:
        iwords.append(wc_dict[item])
    X_list.append(iwords)


# In[16]:


X = np.array(X_list)
y = np.array(y_list)


# In[17]:


maxx = 0
maxs = ""
for index,row in df_1K.iterrows():
    s = row['Summary']
    l = len(s)
    if(l>maxx):
        maxx = l
        maxs = s
print 'Max len in document:',maxx
print maxs


# In[36]:


from keras.utils.np_utils import to_categorical
cy = to_categorical(y, num_classes=None)
#cy


# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, cy, test_size=0.3, stratify = cy)


# In[38]:


# truncate and pad input sequences
max_review_length = 474
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# In[39]:


embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(1675, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(6, activation='sigmoid'))


# In[40]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[41]:


model.fit(X_train, y_train, epochs=10, batch_size=64)


# In[42]:


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[48]:


pred = model.predict_classes(X_test)


# In[49]:


yt = []
for item in y_test:
    ctr = 0
    for i in item:
        if(i==1):
            yt.append(ctr)
        ctr+=1


# In[50]:


pred2 =  model.predict(X_test)


# In[51]:


print metrics.accuracy_score(yt, pred)


# In[52]:


print metrics.classification_report(yt, pred)


# ### ROC Curve

# In[53]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle


# In[54]:


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()


# In[57]:


y_score = pred2


# In[58]:


for i in range(1,6):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[59]:


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# In[60]:


n_classes = 5
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(1,6)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(1,6):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# In[61]:


# Plot all ROC curves
lw = 2
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red','green'])
for i, color in zip(range(1,6), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right", prop={'size': 8})
plt.show()

