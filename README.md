# Collaborative-Based-Movie-Recommendation-System
Recommender systems are the systems which filter information according to a user's choices. It is used in various fields such as books, movies, music, articles etc.
There are two types of recommender systems:
i) Content-Bases recommender system- The main paradigm of a content based recommender system is driven by the statement "Show me more of the same of what I have liked before". It figures out what a user's favourite aspects of an item are and then recommendations on items that share those aspects.
ii) Collaborative-Based recommender system- Collaborative filtering is based on user's saying " Tell me what is popular among my neighbours, I also might like it." It finds similar group of users and provide recommendation based on similar tastes within that group.

In this notebook, we will be using Collaborative based filtering algorithm to recommend movies to the user.
We have taken a MovieLens 20M dataset from kaggle(https://www.kaggle.com/grouplens/movielens-20m-dataset)

We imported the basic libraries that are Numpy, Pandas Matplotlib and sqrt.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

Then we named CSV files according to our prefrences and then displayed movies_csv and ratings_csv by movies.head() and ratings.head()

We then extracted the year of the movie and mentioned it in the different column named year.

movies['year']=movies.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies['year']=movies.year.str.extract('(\d\d\d\d)',expand=False)

movies['title']=movies.title.str.replace('(\(\d\d\d\d\))', '')
movies['title']=movies['title'].apply(lambda x:x.strip())

We dropped the column which is no longer useful that is the "Genre" from movies dataset and "timestamp" from ratings datasets.

movies.drop(columns=['genres'],inplace=True)

ratings.drop(columns=['timestamp'],inplace=True)

Now, we start to work on our recommender system:

The process for creating a User Based recommendation system is as follows:

i)Select a user with the movies the user has watched
ii)Based on his rating to movies, find the top X neighbours
iii)Get the watched movie record of the user for each neighbour.
iv)Calculate a similarity score using some formula
v)Recommend the items with the highest score


Let's begin by creating an input user to recommend movies to:

userInput=[
    {'title':'Breakfast Club, The','rating':5},
    {'title':'Toy Story','rating':3.5},
    {'title':'Jumanji','rating':2},
    {'title':'Pulp Fiction','rating':5},
    {'title':'Akira','rating':4.5}
]

Now, we will find the users who have seen the same movies. With the movie_id in our input, we can get subset of users that have watched or reviewed the same movie.

userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()

We now group the users by user_id
userSubsetGroup = userSubset.groupby(['userId'])

Let's also sort these groups so the users that share the most movies in common with the input have higher priority.

userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)


Next step is we will compare all users to our specified user to find out the one that is more similar.we're going to find out how similar each user is to the input through the Pearson Correlation Coefficient. It is used to measure the strength of a linear association between two variables.
Why Pearson Correlation?

Pearson correlation is invariant to scaling, i.e. multiplying all elements by a nonzero constant or adding any constant to all elements. For example, if you have two vectors X and Y,then, pearson(X, Y) == pearson(X, 2 * Y + 3). This is a pretty important property in recommendation systems because for example two users might rate two series of items totally different in terms of absolute rates, but they would be similar users (i.e. with similar ideas) with similar rates in various scales.
The values given by the formula vary from r = -1 to r = 1, where 1 forms a direct correlation between the two entities (it means a perfect positive correlation) and -1 forms a perfect negative correlation.

In our case, a 1 means that the two users have similar tastes while a -1 means the opposite.
Now, we calculate the Pearson Correlation between input user and subset group, and store it in a dictionary, where the key is the user Id and the value is the coefficient.


pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
   
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
 
    nRatings = len(group)
  
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
   
    tempRatingList = temp_df['rating'].tolist()
   
    tempGroupList = group['rating'].tolist()

    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0
        
pearsonCorrelationDict.items()

Now we will get the top 50 users that are most similar to the input.
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()


Now, we will recommend movies to the input user.
We're going to do this by taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight. But to do this, we first need to get the movies watched by the users in our pearsonDF from the ratings dataframe and then store their correlation in a new column called _similarityIndex". This is achieved below by merging of these two tables.

topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()

Now all we need to do is simply multiply the movie rating by its weight (The similarity index), then sum up the new ratings and divide it by the sum of the weights.

We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId and then dividing two columns:

It shows the idea of all similar users to candidate movies for the input user:

topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()

tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()

#Creates an empty dataframe

recomm_df=pd.DataFrame()
recomm_df['weightedAverageRecommScore']=tempTopUserRating['sum_weightedRating']/tempTopUserRating['sum_similarityIndex']
recomm_df['movieId']=tempTopUserRating.index
recomm_df.head()

Now, let us sort it out and see top 10 movies that algorithm recommended.

recomm_df=recomm_df.sort_values(by='weightedAverageRecommScore',ascending=False)
recomm_df.head(10)

movies.loc[movies['movieId'].isin(recomm_df.head(10)['movieId'].tolist())]

At last, the dataframe will be displayed containing 10 movies that was recommender to the user.

The challenges that I faced during this project was the selection of data to run this collaborative algorithms because it can be done in various ways.

