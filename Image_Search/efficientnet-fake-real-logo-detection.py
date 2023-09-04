#!/usr/bin/env python
# coding: utf-8

# In[13]:


import tensorflow as tf
import numpy as np 
from PIL import Image

model = tf.saved_model.load('C:\\Users\\DGU_ICE\\FindOwn\\Image_Search\\EfficientNet')
classes = [ "Fake" ,  "Genuine" ]


# In[14]:
Id=[]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk(r"C:\Users\DGU_ICE\FindOwn\ImageDB\train"):
    for filename in filenames:
        Id.append(os.path.join(dirname, filename))
Id[:]

train=pd.DataFrame()
train=train.assign(filename=Id)
train.head()


train['label']=train['filename']
train['label']=train['label'].str.replace(r'C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\train\\','')
train.head()


# In[17]:


train['label'] = train['label'].str.split('/').str[0]
train.head()


# In[18]:


Id=[]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\test'):
    for filename in filenames:
        Id.append(os.path.join(dirname, filename))
Id[:]


# In[19]:


test=pd.DataFrame()
test=test.assign(filename=Id)
test.head()


# In[20]:


test['label']=test['filename']
test['label']=test['label'].str.replace(r'C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\test\\','')
test.head()


# In[21]:


test['label'] = test['label'].str.split('\\').str[0]
test.head()


# In[22]:


result=[]
for i in test.filename:
    img = Image.open(i).convert('RGB')
    img = img.resize((300, 300 * img.size[1] // img.size[0]), Image.ANTIALIAS)
    inp_numpy = np.array(img)[None]
    inp = tf.constant(inp_numpy, dtype='float32')
    class_scores = model(inp)[0].numpy()
    result.append(classes[class_scores.argmax()])
result[:]


# In[23]:


test=test.assign(prediction=result)
test.head()


# In[24]:
result=[]
for i in train.filename:
    img = Image.open(i).convert('RGB')
    img = img.resize((300, 300 * img.size[1] // img.size[0]), Image.ANTIALIAS)
    inp_numpy = np.array(img)[None]
    inp = tf.constant(inp_numpy, dtype='float32')
    class_scores = model(inp)[0].numpy()
    result.append(classes[class_scores.argmax()])
result[:]
# In[25]:
train=train.assign(prediction=result)
train.head()

# In[26]:
from sklearn.metrics import classification_report
print(classification_report(train['label'],train['prediction']))


# In[27]:
print(classification_report(test['label'],test['prediction']))

