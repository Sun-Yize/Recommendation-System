# Recommendation System

This repo is for STAT7008 Project. The primary goal of a recommendation system is to provide personalized and targeted recommendations to users, ultimately enhancing their experience and helping them find what they are looking for. These recommendations are generated by analyzing and understanding user data, such as previous actions, preferences, demographic information, and feedback.

### Prerequisite

Unpack data:

```shell
sh script/process.sh
```

Install packages:

```shell
pip install -r requirements.txt
```

### Repo Structure

+ data: movielens 1k and Last.FM dataset
+ src: 
  + dataset: dataloader and preprocess data py
  + model: core model py
  + utils: some useful tools
+ main.ipynb: contain all scripts for three questions.

### Example

Typical recommendation algorithms: Content-based filtering, Item-based collaborative filtering, and User-based collaborative filtering

Data Loading:


```python
datapath = "data/ml-1m/"

# Load Movies
movies = pd.read_csv(datapath + 'movies.dat', delimiter='::', engine='python', header=None, names=['MovieID', 'Title', 'Genres'], encoding='latin-1')

# Load Ratings
ratings = pd.read_csv(datapath + 'ratings.dat', delimiter='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')

# Load Users
users = pd.read_csv(datapath + 'users.dat', delimiter='::', engine='python', header=None, names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='latin-1')
```

Content-Based Filtering: 


```python
# Create a TF-IDF matrix of unigrams, bigrams, and trigrams for each movie's genre
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['Genres'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on the cosine similarity score of movie genres
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # 10 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['Title'].iloc[movie_indices]

# Example usage
recommendations = get_recommendations('Toy Story (1995)')
recommendations
```




    1050            Aladdin and the King of Thieves (1996)
    2072                          American Tail, An (1986)
    2073        American Tail: Fievel Goes West, An (1991)
    2285                         Rugrats Movie, The (1998)
    2286                              Bug's Life, A (1998)
    3045                                Toy Story 2 (1999)
    3542                             Saludos Amigos (1943)
    3682                                Chicken Run (2000)
    3685    Adventures of Rocky and Bullwinkle, The (2000)
    12                                        Balto (1995)
    Name: Title, dtype: object



Item-Based Collaborative Filtering


```python
# Create a pivot table with movies as rows and users as columns
movie_ratings = ratings.pivot_table(index='MovieID', columns='UserID', values='Rating').fillna(0)
item_similarity = cosine_similarity(movie_ratings)

# Function to recommend movies based on item similarity
def get_item_based_recommendation(movie_id):
    # Get movie index for similarity matrix
    idx = movies[movies['MovieID'] == movie_id].index[0]
    similar_scores = item_similarity[idx]
    similar_movies = list(movie_ratings.index[np.where(similar_scores > 0.5)])
    similar_movies.remove(movie_id)  # Remove the movie itself from the recommendation
    return movies[movies['MovieID'].isin(similar_movies)]['Title']

# Example usage
recommendations = get_item_based_recommendation(1)  # For movie with MovieID 1
recommendations[:10]
```




    33                                   Babe (1995)
    38                               Clueless (1995)
    257    Star Wars: Episode IV - A New Hope (1977)
    293                          Pulp Fiction (1994)
    315             Shawshank Redemption, The (1994)
    352                          Forrest Gump (1994)
    360                        Lion King, The (1994)
    453                         Fugitive, The (1993)
    476                         Jurassic Park (1993)
    584                               Aladdin (1992)
    Name: Title, dtype: object



User-Based Collaborative Filtering:


```python
# Create a pivot table with users as rows and movies as columns
user_ratings = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)
user_similarity = cosine_similarity(user_ratings)

# Function to recommend movies based on user similarity
def get_user_based_recommendation(user_id):
    # Get user index for similarity matrix
    idx = users[users['UserID'] == user_id].index[0]
    similar_users = user_similarity[idx]
    similar_users_index = np.where(similar_users > 0.5)[0]
    recommended_movies = set()
    for i in similar_users_index:
        movies_rated_by_similar_user = user_ratings.columns[np.where(user_ratings.iloc[i] > 3)].tolist()
        recommended_movies.update(movies_rated_by_similar_user)
    return movies[movies['MovieID'].isin(recommended_movies)]['Title']

# Example usage
recommendations = get_user_based_recommendation(1)  # For user with UserID 1
list(recommendations)[:10]
```




    ['Toy Story (1995)',
     'Pocahontas (1995)',
     'Apollo 13 (1995)',
     'Star Wars: Episode IV - A New Hope (1977)',
     "Schindler's List (1993)",
     'Secret Garden, The (1993)',
     'Aladdin (1992)',
     'Snow White and the Seven Dwarfs (1937)',
     'Beauty and the Beast (1991)',
     'Fargo (1996)']
