#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pickle


# In[3]:


testmodel=pickle.load(open(r'C:\Users\AJ\Newfolder\Svm_model.sav','rb'))


# In[4]:


input=(1,120,78,20,89,67,102,29)
input_np=np.asarray(input)
input_re=input_np.reshape(1,-1)
# std_=sc.transform(input_re)
pred=testmodel.predict(input_re)
if (pred[0]==1):
    print('diabetes confirmed!!!')
else:
    print('No diabetes confirmed !!!')

