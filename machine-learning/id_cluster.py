#!/usr/bin/env python
# coding: utf-8

# In[1]:


from RecommendationSys_Test2 import cluster
import pandas as pd

data = pd.read_csv('C:\\Users\\HYUNCOM\\Desktop\\연구실\\flask-server\\test_dataset.csv',encoding="cp949")

user_data = cluster(data)

cluster_data = user_data[['user_id', 'cluster']]

print(cluster_data)