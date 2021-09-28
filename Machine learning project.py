#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and dataset

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('laptop_price.csv')
df.head()


# In[4]:


df.shape


# # Cleaning dataset

# In[5]:


df.info()


# In[8]:


df.duplicated().sum()


# In[9]:


df.isnull().sum()


# In[10]:


df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')


# In[11]:


df.head(2)


# # Importing libraries for data visualization 

# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[13]:


sns.distplot(df['Price_euros'])


# In[14]:


df['Company'].value_counts().plot(kind='bar')


# In[15]:


sns.barplot(x=df['Company'],y=df['Price_euros'])
plt.xticks(rotation='vertical')
plt.show()


# In[16]:


df['TypeName'].value_counts().plot(kind='bar')


# In[17]:


sns.barplot(x=df['TypeName'],y=df['Price_euros'])
plt.xticks(rotation='vertical')
plt.show()


# In[18]:


sns.distplot(df['Inches'])


# In[19]:


sns.scatterplot(x=df['Inches'],y=df['Price_euros'])


# In[20]:


df['ScreenResolution'].value_counts()


# In[21]:


df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[22]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[23]:


sns.barplot(x=df['Touchscreen'],y=df['Price_euros'])


# In[24]:


df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[25]:


df['Ips'].value_counts().plot(kind='bar')


# In[26]:


sns.barplot(x=df['Ips'],y=df['Price_euros'])


# In[27]:


new = df['ScreenResolution'].str.split('x',n=1,expand=True)


# In[28]:


new


# In[29]:


df['X_res'] = new[0]
df['Y_res'] = new[1]


# In[30]:


df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[31]:


df.head()


# In[32]:


df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')


# In[33]:


df.corr()['Price_euros']


# In[34]:


df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')


# In[35]:


df.corr()['Price_euros']


# In[36]:


df.drop(columns=['ScreenResolution','laptop_ID','Inches','X_res','Y_res'],inplace=True)


# In[37]:


df['Cpu'].value_counts()


# In[38]:


df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[39]:


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[40]:


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)


# In[41]:


df['Cpu brand'].value_counts().plot(kind='bar')


# In[42]:


sns.barplot(x=df['Cpu brand'],y=df['Price_euros'])
plt.xticks(rotation='vertical')
plt.show()


# In[43]:


df.drop(columns=['Cpu','Cpu Name'],inplace=True)


# In[44]:


df['Ram'].value_counts().plot(kind='bar')


# In[45]:


sns.barplot(x=df['Ram'],y=df['Price_euros'])
plt.xticks(rotation='vertical')
plt.show()


# In[46]:


df['Memory'].value_counts()


# In[79]:


df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


# In[80]:


df.drop(columns=['Memory'],inplace=True)


# In[81]:


df.corr()['Price_euros']


# In[82]:


df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)


# In[83]:


df['Gpu'].value_counts()


# In[84]:


df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])


# In[85]:


df['Gpu brand'].value_counts()


# In[86]:


df = df[df['Gpu brand'] != 'ARM']


# In[87]:


sns.barplot(x=df['Gpu brand'],y=df['Price_euros'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()


# In[88]:


df.drop(columns=['Gpu'],inplace=True)


# In[89]:


df.drop(columns=['Product'],inplace=True)
df.head()


# In[90]:


df['Price'] = df['Price_euros']*87.53
df.drop(['Price_euros'],axis=1,inplace=True)


# In[91]:


df.head()


# In[92]:


df['OpSys'].value_counts()


# In[93]:


sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[94]:


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[95]:


df['os'] = df['OpSys'].apply(cat_os)
df.drop(columns=['OpSys'],inplace=True)


# In[96]:


sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[97]:


sns.distplot(df['Weight'])


# In[98]:


sns.scatterplot(x=df['Weight'],y=df['Price'])


# In[99]:


df.corr()['Price']


# In[100]:


sns.heatmap(df.corr(),annot=True)


# In[101]:


sns.distplot(np.log(df['Price']))


# In[102]:


X = df.drop(columns=['Price'])
y = np.log(df['Price'])


# # Train and Test Split

# In[103]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)


# In[104]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error


# In[111]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# # Linear Regression

# In[112]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Random Forest

# In[113]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Decision Tree

# In[110]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[ ]:




