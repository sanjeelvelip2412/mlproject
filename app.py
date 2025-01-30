from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re

app = Flask(__name__)

# Load the movie dataset with additional fields
movies_df = pd.read_csv('movies.csv')
movies_df = movies_df[['Series_Title', 'Genre', 'Director', 'IMDB_Rating', 'Poster_Link', 'Released_Year', 'Overview', 'Star1', 'Star2']]

# Increase image quality by replacing low-res dimensions in the Poster_Link URL
def enhance_image_quality(url):
    # Use a regex to find and replace the dimensions in the URL
    return re.sub(r'UX\d+_CR0,0,\d+,\d+', 'UX500_CR0,0,500,750', url)  # Change to higher resolution

# Apply the quality enhancement to all poster links
movies_df['Poster_Link'] = movies_df['Poster_Link'].apply(enhance_image_quality)

# Define combined features for content-based filtering, prioritizing Genre and Director
movies_df['combined_features'] = movies_df['Genre'] + ' ' + movies_df['Director'] + ' ' + movies_df['Series_Title']
count_matrix = CountVectorizer().fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(count_matrix)

# Recommendation function for title and genre with partial matches and suggestions
def find_similar_movies(query):
    recommendations = []
    seen_titles = set()  # This will keep track of titles we've already added

    query_lower = query.lower()

    # Find exact or partial title matches
    title_matches = movies_df[movies_df['Series_Title'].str.lower().str.contains(query_lower)]

    # Add exact or partial title matches to recommendations
    for _, title_movie in title_matches.iterrows():
        # Skip if the movie is already in the recommendations
        if title_movie['Series_Title'] in seen_titles:
            continue
        recommendations.append({
            'Series_Title': title_movie['Series_Title'],
            'Genre': title_movie['Genre'],
            'Director': title_movie['Director'],
            'IMDB_Rating': title_movie['IMDB_Rating'],
            'Poster_Link': title_movie['Poster_Link'],
            'Released_Year': title_movie['Released_Year'],
            'Overview': title_movie['Overview'],
            'Star1': title_movie['Star1'],
            'Star2': title_movie['Star2']
        })
        seen_titles.add(title_movie['Series_Title'])  # Mark this movie as seen

    # Find similar movies if there was a title match
    if not title_matches.empty:
        movie_index = title_matches.index[0]
        similarity_scores = list(enumerate(cosine_sim[movie_index]))
        sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]

        for i in sorted_similar_movies:
            similar_movie = movies_df.iloc[i[0]]
            if similar_movie['Series_Title'] in seen_titles:
                continue  # Skip if already added
            recommendations.append({
                'Series_Title': similar_movie['Series_Title'],
                'Genre': similar_movie['Genre'],
                'Director': similar_movie['Director'],
                'IMDB_Rating': similar_movie['IMDB_Rating'],
                'Poster_Link': similar_movie['Poster_Link'],
                'Released_Year': similar_movie['Released_Year'],
                'Overview': similar_movie['Overview'],
                'Star1': similar_movie['Star1'],
                'Star2': similar_movie['Star2']
            })
            seen_titles.add(similar_movie['Series_Title'])  # Mark this movie as seen

    # Handle genre-based suggestions if no title match
    if title_matches.empty:
        genre_matches = movies_df[movies_df['Genre'].str.contains(query, case=False, na=False)]
        for _, genre_movie in genre_matches.head(10).iterrows():
            if genre_movie['Series_Title'] in seen_titles:
                continue  # Skip if already added
            recommendations.append({
                'Series_Title': genre_movie['Series_Title'],
                'Genre': genre_movie['Genre'],
                'Director': genre_movie['Director'],
                'IMDB_Rating': genre_movie['IMDB_Rating'],
                'Poster_Link': genre_movie['Poster_Link'],
                'Released_Year': genre_movie['Released_Year'],
                'Overview': genre_movie['Overview'],
                'Star1': genre_movie['Star1'],
                'Star2': genre_movie['Star2']
            })
            seen_titles.add(genre_movie['Series_Title'])  # Mark this movie as seen

    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendations', methods=['GET'])
def recommend_movies():
    movie_title_or_genre = request.args.get('movie')
    if not movie_title_or_genre:
        return jsonify({"error": "Please provide a movie title or genre."}), 400

    similar_movies = find_similar_movies(movie_title_or_genre)
    return jsonify(similar_movies)

if __name__ == '__main__':
    app.run(debug=True)


