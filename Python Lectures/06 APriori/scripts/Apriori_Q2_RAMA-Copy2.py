#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[19]:


df_encode = pd.DataFrame(encoder_array, columns = encoder.columns_)
frequent_items = apriori(df_encode, min_support = 0.0001, use_colnames = True)
rules = association_rules(frequent_items, metric = 'lift', min_threshold = 1)
rules
def get_index(setList, frequent2):
    ''' Extracts the index '''
    i = -1
    for data in setList:
        i = i + 1
        if data == frequent2:
            return i
    return -1
newindex = []
for idx, value in enumerate(rules['antecedents']):
    if len(value) >= 2:
        if get_index(value, '') != 0:
                  index1.append(idx)


# In[25]:


Index = [value for value in newindex if value in new_index]
rules = rules.loc[index].sort_values(by = ['lift'], ascending = False).head(20)


# In[23]:


filter_rules = rules[rules['confidence'] >= 0.4]
filter_rules = filter_rules[ filter_rules['antecedents']== frozenset({''}) ]
filter_rules = filter_rules.sort_values(by = ['lift'],ascending = False).head(20)
filter_rules

