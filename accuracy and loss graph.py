#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install tensorflow')


# In[3]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[5]:


tf.__version__


# In[6]:


df = pd.read_csv('Churn_Modelling.csv')
df.head()


# In[9]:


dt=df.drop(['RowNumber','CustomerId','Surname','Exited'],axis=1)
dt.head()


# In[31]:


x = dt.iloc[:,:].values
print (x)


# In[13]:


y = df.iloc[:, -1].values
print(y)


# In[22]:





# In[32]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])
print(x)


# In[36]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


# In[38]:


print(x)


# In[40]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)


# In[42]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[43]:


ann = tf.keras.models.Sequential()


# In[44]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# In[45]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# In[46]:


ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[47]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[49]:


ann.fit(x_train, y_train, batch_size = 32, epochs = 100)


# In[51]:


ann.fit(x_train, y_train, batch_size = 32, epochs = 100)


# In[54]:


y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[55]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[77]:


history = ann.fit(x_train, y_train, batch_size = 32, epochs = 100)


# In[83]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.title('training accuracy')
plt.ylabel('tr.accuracy')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper left')
plt.show


# In[79]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper right')
plt.show


# In[80]:


history1 = ann.fit(x_test, y_test, batch_size = 32, epochs = 100)


# In[82]:


import matplotlib.pyplot as plt
plt.plot(history1.history['accuracy'])
plt.title('testing accuracy')
plt.ylabel('ts.accuracy')
plt.xlabel('epoch')
plt.legend(['test'],loc='upper left')
plt.show


# In[ ]:




