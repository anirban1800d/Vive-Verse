from flask import Flask, request, jsonify, render_template
import joblib
import logging
import numpy as np
import requests
 # Contains API keys
import json
import time
from flask import send_from_directory
import os
import random
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "262259b3")
RAWG_API_KEY = os.getenv("RAWG_API_KEY", "c21c769dd6b44b2cbc413a0cff68ee3b")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "b19b9346cf3b4e2aae1800b940380a91")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "e932853e26ab4cba94e48265963db341")




app = Flask(__name__)




pipe_lr = joblib.load("models/text_emotion.pkl")


# Emotion emoji mapping
emotions_emoji_dict = {
    "anger": "😠", 
    "disgust": "🤮", 
    "fear": "😨😱", 
    "happy": "🤗", 
    "joy": "😂", 
    "neutral": "😐", 
    "sad": "😔",
    "sadness": "😔", 
    "shame": "😳", 
    "surprise": "😮"
}

# Emotion-to-category mapping for recommendations
emotion_category_map = {
    "happy": "comedy",
    "joy": "family",
    "neutral": "drama",
    "sad": "romance",
    "anger": "action",
    "disgust": "thriller",
    "fear": "horror",
    "surprise": "fantasy"
}

# Emotion-to-genre mapping for games
emotion_game_map = {
    "happy": "adventure",
    "joy": "indie",
    "neutral": "simulation",
    "sad": "visual-novel",
    "anger": "fighting",
    "disgust": "strategy",
    "fear": "horror",
    "surprise": "rpg"
}

def predict_text_sentiment(text):
    prediction = pipe_lr.predict([text])[0]
    probability = np.max(pipe_lr.predict_proba([text]))
    return prediction, probability


def get_prediction_proba(text):
    """Get the probability distribution for all emotions"""
    results = pipe_lr.predict_proba([text])
    return results[0]

@app.route('/movies', methods=['GET'])
def get_movie_recommendations(emotion, max_results=5):
    """Fetch movie recommendations from OMDb API based on emotion-mapped genre."""
    try:
        # Map emotion to relevant movie genre
        genre = emotion_category_map.get(emotion, "drama")  # Default to 'drama' if no match
        
        # API request
        url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&s={genre}&type=movie"
        session = requests.Session()
        response = session.get(url, timeout=5)  # Set a timeout for network efficiency
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        data = response.json()

        # Check if 'Search' exists in response data
        if "Search" in data and isinstance(data["Search"], list):
            movies = [
                {"title": movie["Title"], "year": movie["Year"], "imdb_id": movie["imdbID"]}
                for movie in data["Search"]  
            ]
            return random.sample(movies, min(len(movies), max_results))

    except requests.exceptions.Timeout:
        print("Movie API Timeout: The request took too long to respond.")
    except requests.exceptions.ConnectionError:
        print("Movie API Connection Error: Unable to connect to OMDb API.")
    except requests.exceptions.HTTPError as e:
        print(f"Movie API HTTP Error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Movie API Error: {e}")
    except KeyError as e:
        print(f"Unexpected data format from OMDb API: {e}")

    return []



@app.route('/games', methods=['GET'])
def get_game_recommendations(emotion):
    try:
        genre = emotion_game_map.get(emotion, "adventure")  # Default to 'adventure'
        url = f"https://api.rawg.io/api/games?key={RAWG_API_KEY}&genres={genre}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if "results" in data:
            games = [game["name"] for game in data["results"]]
            return random.sample(games, min(len(games), 5)) 
    except requests.exceptions.RequestException as e:
        print(f"Game API Error: {e}")
    return []

# Spotify API Authentication
#SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
#SPOTIFY_API_URL = "https://api.spotify.com/v1/recommendations"
#SPOTIFY_CLIENT_ID = "b19b9346cf3b4e2aae1800b940380a91"
#SPOTIFY_CLIENT_SECRET = "e932853e26ab4cba94e48265963db341"

spotify_access_token = None
token_expires_at = 0
@app.route('/spotify_token', methods=['GET'])
def get_spotify_access_token():
    global spotify_access_token, token_expires_at

    if time.time() < token_expires_at:
        return spotify_access_token  # Use existing token if valid

    try:
        auth_url = "https://accounts.spotify.com/api/token"
        auth_data = {
            "grant_type": "client_credentials",
            "client_id": SPOTIFY_CLIENT_ID,
            "client_secret": SPOTIFY_CLIENT_SECRET
        }
        response = requests.post(auth_url, data=auth_data)
        response.raise_for_status()
        token_data = response.json()
        
        spotify_access_token = token_data["access_token"]
        token_expires_at = time.time() + token_data["expires_in"]  # Update expiry time
        return spotify_access_token
    except requests.exceptions.RequestException as e:
        print(f"Spotify Auth Error: {e}")
        return None

# Function to get music recommendations
def get_music_recommendations(emotion):
    try:
        token = get_spotify_access_token()
        if not token:
            return []
        
        headers = {"Authorization": f"Bearer {token}"}
        url = f"https://api.spotify.com/v1/search?q={emotion}&type=track&limit=5"
        response = requests.get(url, headers=headers)
        data = response.json()

        if "tracks" in data:
            tracks = [track["name"] for track in data["tracks"]["items"]]
            return random.sample(tracks, min(len(tracks), 5)) 
    except Exception as e:
        print(f"Music API Error: {e}")
    return []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form["text"]
    emotion, confidence = predict_text_sentiment(text)
    emoji = emotions_emoji_dict.get(emotion, "")
    
    # Get probability distribution for visualization
    probabilities = get_prediction_proba(text)
    emotion_probs = []
    for i, emotion_class in enumerate(pipe_lr.classes_):
        emotion_probs.append({
            "emotion": emotion_class,
            "probability": float(probabilities[i])
        })
    
    # Get recommendations
    movies = get_movie_recommendations(emotion)
    music = get_music_recommendations(emotion)
    games = get_game_recommendations(emotion)

    return jsonify({
        "text": text,
        "emotion": emotion,
        "emoji": emoji,
        "confidence": float(confidence),
        "emotion_probabilities": emotion_probs,
        "movies": movies,
        "music": music,
        "games": games
    })
@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 8000)))