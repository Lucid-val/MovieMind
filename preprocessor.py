import pandas as pd
import ast

def clean_data(movies_df):
    movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    movies_df.dropna(inplace=True)


    def parse_genres(obj):
        genres = []
        for item in ast.literal_eval(obj):
            genres.append(item['name'])
        return genres
    
    def parse_keywords(obj):
        keywords = []
        for item in ast.literal_eval(obj):
            keywords.append(item['name'])
        return keywords
    
    def parse_cast(obj):
        cast_list = []
        for item in ast.literal_eval(obj):
            cast_list.append(item['name'])
            if len(cast_list) == 3:
                break
        return cast_list
    
    def parse_crew(obj):
        for item in ast.literal_eval(obj):
            if item['job'] == 'Director':
                return [item['name']]
        return []
    

    movies_df['genres'] =  movies_df['genres'].apply(parse_genres)
    movies_df['keywords'] = movies_df['keywords'].apply(parse_keywords)
    movies_df['cast'] = movies_df['cast'].apply(parse_cast)
    movies_df['crew'] = movies_df['crew'].apply(parse_crew)

    return movies_df

