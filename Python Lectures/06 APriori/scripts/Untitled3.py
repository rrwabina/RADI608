#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
df=pd.read_csv("Downloads/PrescriptionDB.csv")
df.set_index('Item ID')
df.fillna('', inplace = True)
df = df[['Code.1', 'Code.2', 'Code.3', 'Code.4']]
df = df.to_numpy()

dataset = []
for index in range(0, df.shape[0]):
    new_list = list(filter(None, df[index]))
    dataset.append(new_list)
    
encoder = TransactionEncoder()
encoder_array = encoder.fit(dataset).transform(dataset)
encoder_array


# In[86]:


df_encode = pd.DataFrame(encoder_array, columns = encoder.columns_)
frequent_items = apriori(df_encode, min_support = 0.001, use_colnames = True)


# In[88]:


rules = association_rules(frequent_items, metric = 'lift', min_threshold = 1)
rules


# In[109]:


filter_rules = rules[rules['confidence'] >= 0.4]
filter_rules[ filter_rules['consequents'] == frozenset({'OMPZ'}) ]
filter_rules = filter_rules.sort_values(by = ['lift'],ascending = False).head(20)
filter_rules


# In[116]:


print(dataset)

