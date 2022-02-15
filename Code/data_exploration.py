#******************Exploration of movielens_dataset(1M_version)******************#
import pandas as pd
import numpy as np

ratings_loc = 'data/ratings.csv'
movies_loc = 'data/movies.csv'
links_loc = 'data/links.csv'
tags_loc = 'data/tags.csv'

ratings = pd.read_csv(ratings_loc, sep=',')
movies = pd.read_csv(movies_loc, sep=',')
links = pd.read_csv(links_loc, sep=',')
tags = pd.read_csv(tags_loc, sep=',')

#print(ratings.describe())
#print(ratings.head(10))

#ratings: userid, movieid, rating, timestamp
#movies: movieid, title, genre
#links: movieid, imdbid, tmdbid
#tags: userid, movieid, tag, timestamp
#NOTICE: Because of the lack of relationships between users, We may focus on the similarities between movies more.

#******************Create_smaller_dataset******************#
ratings_1 = ratings[ratings['movieId'] <= 100]
ratings_1 = ratings_1[ratings_1['userId'] <= 30]
ratings_1.to_csv('data/ratings_mini.csv', sep=',', header=True, index = False)
#print(ratings_1.describe())

ratings_1 = ratings[ratings['movieId'] <= 100]
ratings_2 = ratings_1.groupby('userId',as_index=False).count().sort_values(by=['movieId'],ascending=False)
ratings_2 = ratings_2.head(30)
namelist = []
for index,row in ratings_2.iterrows():
    namelist.append(row['userId'])
ratings_3 = ratings_1.loc[ratings_1['userId'].isin(namelist)]
ratings_3.to_csv('data/ratings_mini_1.csv', sep=',', header=True, index = False)

ratings_2 = ratings.groupby('movieId',as_index=False).count().sort_values(by=['userId'],ascending=False)
ratings_2 = ratings_2.head(100)
movielist = []
for index,row in ratings_2.iterrows():
    movielist.append(row['movieId'])
ratings_3 = ratings.loc[ratings['movieId'].isin(movielist)]

ratings_2 = ratings_3.groupby('userId',as_index=False).count().sort_values(by=['movieId'],ascending=False)
ratings_2 = ratings_2.head(30)
namelist = []
for index,row in ratings_2.iterrows():
    namelist.append(row['userId'])
ratings_4 = ratings_3.loc[ratings_3['userId'].isin(namelist)]
ratings_4.to_csv('data/ratings_mini_2.csv', sep=',', header=True, index = False)

#******************************************************************************************************************#
'''
ratings_5 = ratings.groupby('userId',as_index=False).count().sort_values(by=['movieId'],ascending=False)
ratings_5 = ratings_5.head(30)
namelist = []
for index,row in ratings_5.iterrows():
    namelist.append(row['userId'])
ratings_6 = ratings.loc[ratings['userId'].isin(namelist)]

ratings_5 = ratings_6.groupby('movieId',as_index=False).count().sort_values(by=['userId'],ascending=False)
ratings_5 = ratings_5.head(100)
movielist = []
for index,row in ratings_2.iterrows():
    movielist.append(row['movieId'])
ratings_7 = ratings_6.loc[ratings_6['movieId'].isin(movielist)]
ratings_7.to_csv('data/ratings_mini_3.csv', sep=',', header=True, index = False)
print(ratings_7.describe())
'''

#******************************************************************************************************************#
def process(frame):
    return list(frame['genres'].split('|'))

movies['genres'] = movies.apply(lambda x:process(x),axis=1)
movies.to_csv('data/new_moveis.csv', sep=',', header=True, index = False)