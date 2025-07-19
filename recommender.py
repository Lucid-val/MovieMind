import pandas as pd
import ast

def load_data():

    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')

    print("Movies dataset shape: ", movies.shape)
    print("Credits dataset shape: ", credits.shape)

    merged = movies.merge(credits, left_on='title', right_on='title')

    print("Merged dataset shape: ", merged.shape)
    print("Sample columns: ", merged.columns.tolist())
    print("Sample row: ", merged.iloc[0])

    return merged

def preprocess_data(movie_df, credits_df):

    movies_df = movies_df.merge(credits_df, on='title')

    movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    movies_df.dropna(inplace=True)


    def parse(obj_str):
        try:
            return [item['name'] for item in ast.literal_eval(obj_str)]
        except:
            return []
        
    movies_df['genres'] = movies_df['genres'].apply(parse)
    movies_df['keywords'] = movies_df['keywords'].apply(parse)

    def extract_top_actors(cast_str):
        try:
            cast = ast.literal_eval(cast_str)
            return [actor['name'] for actor in cast[:3]]
        except:
            return []
        
    movies_df['cast'] = movies_df['cast'].apply(extract_top_actors)


    def get_director(crew_str):
        try:
            crew = ast.literal_eval(crew_str)
            for person in crew:
                if person['job'] == 'Director':
                    return [person['name']]
            return []
        except:
            return []
        
    movies_df['director'] = movies_df['crew'].apply(get_director)


    movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']
    movies_df['tags'] = movies_df['tags'].apply(lambda x: " ".join(x).lower())

    final_df = movies_df[['movie_id', 'title', 'tags']]


    return final_df


if __name__ == '__main__':
    from load_data import load_data  # Assuming you saved it in a separate file; otherwise remove this line
    movies_df, credits_df = load_data() 
    final_df = preprocess_data(movies_df, credits_df)
    print(final_df.head(5))