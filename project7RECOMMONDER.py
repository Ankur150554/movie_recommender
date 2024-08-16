import numpy as np
import pandas as pd
import ast
df=pd.read_csv('C:\\Users\\naval\\Downloads\\archive (1)\\tmdb_5000_movies.csv')
df1=pd.read_csv('C:\\Users\\naval\\Downloads\\archive (1)\\tmdb_5000_credits.csv')
#print(df.columns,df1.columns)
df=df.merge(df1,on='title')
print(df.info())
df=df[['movie_id','title','overview','genres','keywords','cast','crew']]

df.dropna(inplace=True)
#print(df.isnull().sum())

#apllying transformation of rows

def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])

    return l    

df['genres']=df['genres'].apply(convert)
df['keywords']=df['keywords'].apply(convert)
def convert3(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            l.append(i['name'])
            counter+=1
        else:
            break    
    return l    

df['cast']=df['cast'].apply(convert3)
#print(df['cast'])

def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l  

df['crew']=df['crew'].apply(fetch_director)
#print(df['crew'])

df['overview']=df['overview'].apply(lambda x:x.split())
#print(df['overview'])

# we remove spaces from strings 
df['genres']=df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
df['keywords']=df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
df['cast']=df['cast'].apply(lambda x:[i.replace(" ","") for i in x])
df['crew']=df['crew'].apply(lambda x:[i.replace(" ","") for i in x])

df['tag']=df['cast']+df['crew']+df['genres']+df['keywords']+df['overview']
new_df=df[['tag','title','movie_id']]

# Join the list elements into a single string before applying lower()
new_df['tag'] = new_df['tag'].apply(lambda x: " ".join(x).lower())


#do vectorisation 
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')

vectors=cv.fit_transform(new_df['tag']).toarray()
#print(cv.get_feature_names_out())


import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    if isinstance(text, list):
        text = " ".join(text)
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

 

new_df['tag']=new_df['tag'].apply(stem)
#cosine distance 
from sklearn.metrics.pairwise import cosine_similarity
similerty=cosine_similarity(vectors)
print(similerty)

def recommond(movie):
    
    if movie not in new_df['title'].values:
        print(f"Movie '{movie}' not found in the dataset.")
        return
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similerty[movie_index]
    movie_lst=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_lst:
        print(new_df.iloc[i[0]].title)
    
recommond('movie name')

