import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load raw CSVs and merge them
def load_data():
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    return movies.merge(credits, on='title')

# Clean and extract relevant fields
def clean_data(df):
    df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    df.dropna(inplace=True)

    def parse_genres(obj):
        return [item['name'] for item in ast.literal_eval(obj)]

    def parse_keywords(obj):
        return [item['name'] for item in ast.literal_eval(obj)]

    def parse_cast(obj):
        return [item['name'] for item in ast.literal_eval(obj)][:3]

    def parse_crew(obj):
        for item in ast.literal_eval(obj):
            if item['job'] == 'Director':
                return [item['name']]
        return []

    df['genres'] = df['genres'].apply(parse_genres)
    df['keywords'] = df['keywords'].apply(parse_keywords)
    df['cast'] = df['cast'].apply(parse_cast)
    df['crew'] = df['crew'].apply(parse_crew)

    return df

# Create unified tags field
def create_tags_column(df):
    df['overview'] = df['overview'].apply(lambda x: x.split())
    df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
    df['tags'] = df['tags'].apply(lambda x: " ".join(x).lower())
    return df[['movie_id', 'title', 'tags']]

# Vectorize tags
def vectorize_tags(df, max_features=5000):
    cv = CountVectorizer(max_features=max_features, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    return vectors, cv

# Recommendation logic
def recommend(movie_title, df, vectors):
    if movie_title not in df['title'].values:
        return []

    index = df[df['title'] == movie_title].index[0]
    distances = cosine_similarity(vectors[index].reshape(1, -1), vectors).flatten()

    similar_indices = distances.argsort()[::-1][1:6]
    return df.iloc[similar_indices]['title'].values.tolist()
