from flask import Flask, request, jsonify, render_template, redirect
import requests
import cv2
import pickle
import mediapipe as mp
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import face_recognition
import os
import firebase_admin
from firebase_admin import credentials, db
import cloudinary
import cloudinary.uploader
import base64
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask_cors import CORS
import re
from datetime import datetime, timedelta
import time
import uuid
import threading
from dotenv import load_dotenv  # Import python-dotenv

# Load environment variables from api.env
load_dotenv('api.env')

app = Flask(__name__)

# Stopwatch state
stopwatch_running = False
stopwatch_start_time = None
stopwatch_elapsed_time = 0  # To store elapsed time when paused

# Load configurations from environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = os.getenv("NEWS_API_URL")

STOCK_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
STOCK_URL = os.getenv("ALPHA_VANTAGE_API_URL")

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_URL = os.getenv("WEATHER_API_URL")

API_URL = os.getenv("OPENROUTER_API_URL")
HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json"
}

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
SCOPE = "user-read-playback-state user-modify-playback-state streaming user-read-email user-read-private"

FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL")

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Initialize Spotify OAuth manager
sp_oauth = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=SCOPE
)

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=sp_oauth)

# Load the trained model for sign detection
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Initialize Firebase Realtime Database
cred = credentials.Certificate(FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred, {
    'databaseURL': FIREBASE_DATABASE_URL
})
db_ref = db.reference('faces_dataset')
db = db.reference()  # Root reference for tasks

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

# Labels dictionary for sign detection
labels_dict = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H',
    9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O',
    16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V',
    23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: '1', 28: '2', 29: '3',
    30: '4', 31: '5', 32: '6', 33: '7', 34: '8'
}

# Function to fetch chatbot response
def get_response_from_api(user_input):
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": user_input}]
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"] if "choices" in data else "I couldn't generate a response."
    except Exception as e:
        return f"Error: {str(e)}"

# Spotify Token Endpoint
@app.route("/get_spotify_token", methods=["GET"])
def get_spotify_token():
    try:
        token_info = sp_oauth.get_cached_token()
        print("Current token info:", token_info) 
        if not token_info:
            print("No token found, redirecting to Spotify auth")
            auth_url = sp_oauth.get_authorize_url()
            return jsonify({"auth_url": auth_url})
        if sp_oauth.is_token_expired(token_info):
            print("Token expired, refreshing")
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            print("Refreshed token:", token_info)
        print("Returning access token:", token_info["access_token"])
        return jsonify({"access_token": token_info["access_token"]})
    except Exception as e:
        print("Error in get_spotify_token:", str(e))
        return jsonify({"error": str(e)}), 500

# Spotify Callback Route
@app.route("/peter", methods=["GET"])
def peter():
    code = request.args.get('code')
    error = request.args.get('error')  # Spotify auth errors (e.g., user denied access)
    
    if error:
        error_desc = request.args.get('error_description', 'No details provided')
        print(f"Spotify OAuth Error: {error} - {error_desc}")
        return render_template("peter.html", error=f"Spotify error: {error_desc}")

    if code:
        try:
            token_info = sp_oauth.get_access_token(code)
            print("Spotify token received:", token_info)
            return redirect("http://127.0.0.1:5000/")
        except Exception as e:
            print("Error during token exchange:", str(e))
            return render_template("peter.html", 
                                 error="Failed to authenticate with Spotify. Please try again.")
    return render_template("peter.html")

# Spotify Play Song Route
@app.route("/play_song", methods=["POST"])
def play_song():
    try:
        data = request.json
        command = data.get('song_name', '').strip().lower()
        use_device_playback = data.get('use_device_playback', False)

        if not command:
            return jsonify({"error": "Song command is required"}), 400

        song_name = None
        artist_name = None
        movie_name = None

        patterns = [
            r"play\s+(.+?)\s+by\s+(.+?)\s+from\s+(.+)",  # "Play [song] by [artist] from [movie]"
            r"play\s+(.+?)\s+from\s+(.+)",              # "Play [song] from [movie]"
            r"play\s+(.+?)\s+by\s+(.+)",                # "Play [song] by [artist]"
            r"play\s+(.+?)\s+song"                      # "Play [song] song"
        ]

        for pattern in patterns:
            match = re.match(pattern, command)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    song_name, artist_name, movie_name = groups
                elif len(groups) == 2:
                    if "from" in pattern:
                        song_name, movie_name = groups
                    else:
                        song_name, artist_name = groups
                elif len(groups) == 1:
                    song_name = groups[0]
                break

        if not song_name:
            return jsonify({"error": "Could not parse song command"}), 400

        query = f"track:{song_name}"
        if artist_name:
            query += f" artist:{artist_name}"
        if movie_name:
            query += f" album:{movie_name}"

        results = sp.search(q=query, type='track', limit=5, market='US')
        tracks = results['tracks']['items']
        if not tracks:
            return jsonify({"response": f"No results found for '{command}'"}), 404

        if artist_name:
            exact_match = None
            for track in tracks:
                track_artists = [artist['name'].lower() for artist in track['artists']]
                if artist_name in track_artists and track['name'].lower() == song_name:
                    exact_match = track
                    break
            if exact_match:
                track_uri = exact_match['uri']
                track_name = exact_match['name']
                artist_name = exact_match['artists'][0]['name']
                thumbnail = exact_match['album']['images'][0]['url'] if exact_match['album']['images'] else None

                if use_device_playback:
                    devices = sp.devices()
                    if not devices['devices']:
                        return jsonify({
                            "response": "No active Spotify device found",
                            "action_required": "Please open Spotify on a device first"
                        }), 404
                    sp.start_playback(uris=[track_uri])
                else:
                    return jsonify({
                        "response": f"Ready to play {track_name} by {artist_name}",
                        "uri": track_uri,
                        "name": track_name,
                        "artist": artist_name,
                        "thumbnail": thumbnail,
                        "playing": True
                    })
                return jsonify({
                    "response": f"Playing {track_name} by {artist_name}",
                    "uri": track_uri,
                    "name": track_name,
                    "artist": artist_name,
                    "thumbnail": thumbnail,
                    "playing": True
                })

        if len(tracks) == 1:
            track = tracks[0]
            track_uri = track['uri']
            track_name = track['name']
            artist_name = track['artists'][0]['name']
            thumbnail = track['album']['images'][0]['url'] if track['album']['images'] else None

            if use_device_playback:
                devices = sp.devices()
                if not devices['devices']:
                    return jsonify({
                        "response": "No active Spotify device found",
                        "action_required": "Please open Spotify on a device first"
                    }), 404
                sp.start_playback(uris=[track_uri])
            else:
                return jsonify({
                    "response": f"Ready to play {track_name} by {artist_name}",
                    "uri": track_uri,
                    "name": track_name,
                    "artist": artist_name,
                    "thumbnail": thumbnail,
                    "playing": True
                })
            return jsonify({
                "response": f"Playing {track_name} by {artist_name}",
                "uri": track_uri,
                "name": track_name,
                "artist": artist_name,
                "thumbnail": thumbnail,
                "playing": True
            })

        options = [
            {
                "uri": track['uri'],
                "name": track['name'],
                "artist": track['artists'][0]['name'],
                "album": track['album']['name'],
                "thumbnail": track['album']['images'][0]['url'] if track['album']['images'] else None
            }
            for track in tracks
        ]
        return jsonify({
            "response": f"Multiple songs found for '{song_name}'. Please select one.",
            "options": options,
            "multiple": True
        })

    except Exception as e:
        print(f"Playback error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}", "response": "Failed to play the song due to an unexpected error."}), 500

# Spotify Stop Song Route
@app.route("/stop_song", methods=["POST"])
def stop_song():
    try:
        sp.pause_playback()
        return jsonify({"response": "Song paused.", "speak": True})
    except Exception as e:
        print("Error in stop_song:", str(e))
        return jsonify({"response": f"Error stopping song: {str(e)}", "speak": True})

# Spotify List Songs Route
@app.route("/list_songs", methods=["GET"])
def list_songs():
    try:
        results = sp.featured_playlists(limit=5)
        playlist_names = [playlist['name'] for playlist in results['playlists']['items']]
        response_text = f"Featured playlists: {', '.join(playlist_names)}"
        return jsonify({"response": response_text, "speak": True})
    except Exception as e:
        print("Error in list_songs:", str(e))
        return jsonify({"response": f"Error listing songs: {str(e)}", "speak": True})

@app.route("/set_volume", methods=["POST"])
def set_volume():
    volume = request.json.get('volume', 100)
    try:
        sp.volume(int(volume))
        return jsonify({"response": "Volume set"})
    except Exception as e:
        print("Error in set_volume:", str(e))
        return jsonify({"response": f"Error setting volume: {str(e)}"})

# Sign Detection Route
@app.route("/detect_sign", methods=["POST"])
def detect_sign():
    cap = cv2.VideoCapture(0)
    detected_sign = None
    try:
        ret, frame = cap.read()
        if not ret:
            return jsonify({"response": "No sign detected! Try once again!", "speak": True})
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            data_aux = []
            x_, y_ = [], []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                    data_aux.append(lm.x)
                    data_aux.append(lm.y)
            x_min, x_max = min(x_), max(x_)
            y_min, y_max = min(y_), max(y_)
            data_aux_normalized = [(x - x_min, y - y_min) for x, y in zip(x_, y_)]
            data_flattened = [coord for pair in data_aux_normalized for coord in pair]
            data_padded = pad_sequences([data_flattened], maxlen=84, padding='post', dtype='float32')[0]
            if len(data_padded) == model.n_features_in_:
                prediction = model.predict([data_padded])
                detected_sign = labels_dict.get(int(prediction[0]), None)
    except Exception as e:
        return jsonify({"response": "Error processing sign detection.", "speak": True})
    finally:
        cap.release()
    if detected_sign:
        return jsonify({"sign": detected_sign})
    else:
        return jsonify({"response": "No sign detected! Try once again!", "speak": True})

# Home Page Route
@app.route("/")
def index():
    return render_template("index.html")

# Avatar Routes
@app.route("/ava")
def ava():
    return render_template("ava.html")

@app.route("/jessica")
def jessica():
    return render_template("jessica.html")

@app.route("/kevin")
def kevin():
    return render_template("kevin.html")

# Chat API Route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip().lower()
    if not user_input:
        return jsonify({"response": "Please say something.", "speak": True})

    task_pattern = r"add task\s+(.+?)\s+at\s+(.+)"  # e.g., "add task Meeting at 14:30"
    relative_task_pattern = r"add task\s+(.+?)\s+in\s+(\d+)\s+(minute|minutes|hour|hours)"  # e.g., "add task Call in 5 minutes"
    sign_detection_pattern = r"(detect sign|read my sign|what sign am i making)"  # Commands to trigger sign detection

    task_match = re.match(task_pattern, user_input)
    relative_task_match = re.match(relative_task_pattern, user_input)
    sign_detection_match = re.search(sign_detection_pattern, user_input)

    if task_match:
        title, time_input = task_match.groups()
        return add_task_from_chat(title, time_input)
    elif relative_task_match:
        title, amount, unit = relative_task_match.groups()
        time_input = f"in {amount} {unit}"
        return add_task_from_chat(title, time_input)
    elif sign_detection_match:
        sign_result = detect_sign()
        sign_data = sign_result.get_json()
        if "sign" in sign_data:
            response = f"You signed the letter {sign_data['sign']}."
            return jsonify({"response": response, "sign": sign_data["sign"], "speak": True})
        return jsonify({"response": sign_data.get("response", "No sign detected."), "speak": True})

    response_text = get_response_from_api(user_input)
    return jsonify({"response": response_text, "speak": True})

# Helper function to add task from chat
def add_task_from_chat(title, time_input):
    try:
        now = datetime.now()
        task_time = None

        if "in " in time_input.lower():
            match = re.search(r"in (\d+) (minute|minutes|hour|hours)", time_input.lower())
            if match:
                amount = int(match.group(1))
                unit = match.group(2)
                if "minute" in unit:
                    task_time = (now + timedelta(minutes=amount)).strftime("%H:%M")
                elif "hour" in unit:
                    task_time = (now + timedelta(hours=amount)).strftime("%H:%M")
            else:
                return jsonify({"response": "Invalid time format. Use 'in X minutes' or 'HH:MM'."})
        else:
            time_input = time_input.replace(" ", "").lower()
            if re.match(r"^\d{4}$", time_input):
                task_time = f"{time_input[:2]}:{time_input[2:]}"
            else:
                for fmt in ["%I:%M%p", "%H:%M", "%I%p"]:
                    try:
                        task_time = datetime.strptime(time_input, fmt).strftime("%H:%M")
                        break
                    except ValueError:
                        continue
            
            if not task_time:
                return jsonify({"response": "Invalid time format. Use HH:MM, H:MM PM, or 'in X minutes'"})

        task_id = str(uuid.uuid4())
        task_data = {
            "id": task_id,
            "title": title,
            "time": task_time,
            "status": "pending",
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        db.child("tasks").child(task_id).set(task_data)
        return jsonify({"response": f"Task '{title}' added for {task_time}.", "speak": True})
    except Exception as e:
        return jsonify({"response": f"Error adding task: {str(e)}", "speak": True})

# Sign Detection API Route
@app.route("/chatbot-response", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "No sign detected."})
    predefined_responses = {char: f"You signed the letter {char}." for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    predefined_responses.update({str(num): f"You signed the number {num}." for num in range(10)})
    response = predefined_responses.get(user_input, get_response_from_api(user_input))
    return jsonify({"response": response})

# Face Recognition Class
class RealTimeFaceRecognition:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.load_known_faces_from_firebase()

    def load_known_faces_from_firebase(self):
        users = db_ref.get()
        if users:
            self.known_faces = []
            self.known_names = []
            for user_id, user_data in users.items():
                if 'face_encoding' in user_data and 'name' in user_data:
                    encoding = np.frombuffer(base64.b64decode(user_data['face_encoding']), dtype=np.float64)
                    self.known_faces.append(encoding)
                    self.known_names.append(user_data['name'])
            print("Loaded known faces from Firebase Realtime Database.")
        else:
            print("No users found in Firebase Realtime Database.")

    def upload_to_cloudinary(self, image, name):
        _, img_encoded = cv2.imencode('.jpg', image)
        result = cloudinary.uploader.upload(img_encoded.tobytes(), public_id=name)
        return result['secure_url']

    def save_to_firebase(self, name, image_url, encoding):
        user_data = {
            'name': name,
            'image_url': image_url,
            'face_encoding': base64.b64encode(encoding.tobytes()).decode('utf-8')
        }
        new_user_ref = db_ref.push(user_data)
        return new_user_ref.key

    def recognize_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        recognized_faces = []
        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_faces, encoding)
            name = "Unknown"
            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(self.known_faces, encoding))
                name = self.known_names[best_match_index]
            recognized_faces.append((name, (top, right, bottom, left), encoding))
        return recognized_faces

recognition = RealTimeFaceRecognition()

def get_bot_response(user_input):
    user_input = user_input.lower()

    if "news" in user_input:
        category = "overall"
        if "sports" in user_input:
            category = "sports"
        elif "business" in user_input:
            category = "business"
        elif "politics" in user_input:
            category = "politics"
        elif "entertainment" in user_input:
            category = "entertainment"
        elif "health" in user_input:
            category = "health"
        elif "science" in user_input:
            category = "science"
        elif "technology" in user_input:
            category = "technology"
        response = requests.get(f"http://127.0.0.1:5000/get_news?category={category}")
        return response.json()

    elif "stock" in user_input:
        symbol = "AAPL"  # Extract from user input if needed
        response = requests.get(f"http://127.0.0.1:5000/get_stock?symbol={symbol}")
        return response.json()

    elif "weather" in user_input:
        city = "New York"  # Extract from user input if needed
        response = requests.get(f"http://127.0.0.1:5000/get_weather?city={city}")
        return response.json()

    else:
        return "I don't have that information."

# Face Recognition API Route
@app.route("/detect_face", methods=["POST"])
def detect_face():
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            print("Error: Camera could not be opened in /detect_face")
            return jsonify({"response": "Unable to access camera.", "speak": True}), 500

        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture image in /detect_face")
            return jsonify({"response": "Unable to capture image.", "speak": True}), 500

        recognition.load_known_faces_from_firebase()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not face_encodings:
            print("No face detected in /detect_face")
            return jsonify({"response": "No face detected. Please show your face.", "speak": True})

        recognized_faces = recognition.recognize_faces(frame)
        faces = []
        for name, _, encoding in recognized_faces:
            if name != "Unknown":
                distances = face_recognition.face_distance(recognition.known_faces, encoding)
                best_match_index = np.argmin(distances)
                best_match_distance = distances[best_match_index]
                if best_match_distance < 0.4:
                    faces.append({"name": name, "new_user": False})
                else:
                    print(f"Face match for {name} was not confident enough (distance: {best_match_distance}). Treating as new user.")
                    faces.append({"name": "Unknown", "new_user": True})
            else:
                faces.append({"name": "Unknown", "new_user": True})

        return jsonify({"faces": faces})

    except Exception as e:
        print(f"Error in detect_face: {str(e)}")
        return jsonify({"response": f"Error: {str(e)}", "speak": True}), 500
    finally:
        cap.release()
        print("Camera released in /detect_face")

# Route to Register New User
@app.route("/register_user", methods=["POST"])
def register_user():
    data = request.get_json()
    name = data.get("name", "").strip()
    encoding_base64 = data.get("encoding", None)
    frame_base64 = data.get("frame", None)

    if not name:
        return jsonify({"response": "Name is required.", "speak": True}), 400
    if not encoding_base64 or not frame_base64:
        return jsonify({"response": "Face data is required.", "speak": True}), 400

    try:
        encoding = np.frombuffer(base64.b64decode(encoding_base64), dtype=np.float64)
        frame_data = base64.b64decode(frame_base64)
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"response": "Invalid frame data.", "speak": True}), 400

        cloudinary_url = recognition.upload_to_cloudinary(frame, name)
        recognition.save_to_firebase(name, cloudinary_url, encoding)
        recognition.known_faces.append(encoding)
        recognition.known_names.append(name)

        return jsonify({"response": f"Welcome, {name}! You have been registered.", "speak": True}), 200

    except Exception as e:
        print(f"Error in register_user: {str(e)}")
        return jsonify({"response": f"Error registering user: {str(e)}", "speak": True}), 500

@app.route("/capture_face", methods=["POST"])
def capture_face():
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            print("Error: Camera could not be opened in /capture_face")
            return jsonify({"success": False, "error": "Unable to access camera."}), 500

        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture image in /capture_face")
            return jsonify({"success": False, "error": "Unable to capture image."}), 500

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)

        if not face_encodings:
            print("No face detected in /capture_face")
            return jsonify({"success": False, "error": "No face detected in the captured frame."}), 400

        encoding = face_encodings[0]
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        print("Face captured successfully in /capture_face")
        return jsonify({
            "success": True,
            "encoding": base64.b64encode(encoding.tobytes()).decode('utf-8'),
            "frame": img_base64
        }), 200

    except Exception as e:
        print(f"Error in capture_face: {str(e)}")
        return jsonify({"success": False, "error": f"Error capturing face: {str(e)}"}), 500
    finally:
        cap.release()
        print("Camera released in /capture_face")

# News Route
@app.route('/get_news', methods=['GET'])
def get_news():
    category = request.args.get("category", "overall").lower()
    try:
        if category == "overall":
            url = f"{NEWS_URL}?apikey={NEWS_API_KEY}&country=in&language=en"
        else:
            url = f"{NEWS_URL}?apikey={NEWS_API_KEY}&country=in&language=en&category={category}"
        
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            return jsonify({"response": f"Error fetching news: Status code {response.status_code}", "speak": True})

        if "results" in data and data["results"]:
            news_items = [f"{i+1}. {article['title']} (Source: {article.get('source_id', 'Unknown')}) - {article['link']}" 
                          for i, article in enumerate(data["results"][:5])]
            response_text = f"Top 5 {category.capitalize()} News Articles:\n" + "\n".join(news_items)
            return jsonify({"response": response_text, "speak": True})
        else:
            return jsonify({"response": f"No news articles found for category: {category}. Try another category.", "speak": True})
    except Exception as e:
        return jsonify({"response": f"Error fetching news: {str(e)}", "speak": True})

@app.route('/get_stock', methods=['GET'])
def get_stock():
    symbol = request.args.get("symbol", "AAPL")
    url = f"{STOCK_URL}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={STOCK_API_KEY}"
    response = requests.get(url)
    data = response.json()
    return jsonify(data)

@app.route('/get_weather', methods=['GET'])
def get_weather():
    city = request.args.get("city", "New York")
    url = f"{WEATHER_URL}?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return jsonify(data)

# Task Scheduler Routes
@app.route("/add-task", methods=["POST"])
def add_task():
    try:
        data = request.get_json()
        title = data.get("title")
        time_input = data.get("time")
        
        if not title or not time_input:
            return jsonify({"success": False, "error": "Missing title or time"})

        now = datetime.now()
        task_time = None

        if "in " in time_input.lower():
            match = re.search(r"in (\d+) (minute|minutes|hour|hours)", time_input.lower())
            if match:
                amount = int(match.group(1))
                unit = match.group(2)
                if "minute" in unit:
                    task_time = (now + timedelta(minutes=amount)).strftime("%H:%M")
                elif "hour" in unit:
                    task_time = (now + timedelta(hours=amount)).strftime("%H:%M")
            else:
                return jsonify({"success": False, "error": "Invalid relative time format"})
        else:
            time_input = time_input.replace(" ", "").lower()
            if re.match(r"^\d{4}$", time_input):
                task_time = f"{time_input[:2]}:{time_input[2:]}"
            else:
                for fmt in ["%I:%M%p", "%H:%M", "%I%p"]:
                    try:
                        task_time = datetime.strptime(time_input, fmt).strftime("%H:%M")
                        break
                    except ValueError:
                        continue
            
            if not task_time:
                return jsonify({"success": False, "error": "Invalid time format. Use HH:MM, H:MM PM, or 'in X minutes'"})

        task_id = str(uuid.uuid4())
        task_data = {
            "id": task_id,
            "title": title,
            "time": task_time,
            "status": "pending",
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        db.child("tasks").child(task_id).set(task_data)
        return jsonify({"success": True, "task": task_data})
    except Exception as e:
        return jsonify({"success": False, "error": f"Error adding task: {str(e)}"})

@app.route("/get-tasks", methods=["GET"])
def get_tasks():
    try:
        tasks = db.child("tasks").get().val() or {}
        task_list = [{**task, "id": task_id} for task_id, task in tasks.items()]
        return jsonify({"success": True, "tasks": task_list})
    except Exception as e:
        return jsonify({"success": False, "error": f"Error fetching tasks: {str(e)}"})

@app.route("/delete-task", methods=["POST"])
def delete_task():
    try:
        data = request.get_json()
        task_id = data.get("taskId")
        if not task_id:
            print("Delete task failed: Missing taskId")
            return jsonify({"success": False, "error": "Missing taskId"})
        
        task = db.child("tasks").child(task_id).get().val()
        if not task:
            print(f"Delete task failed: Task {task_id} not found")
            return jsonify({"success": False, "error": "Task not found"})
        
        db.child("tasks").child(task_id).remove()
        print(f"Task deleted: {task['title']} (ID: {task_id})")
        return jsonify({"success": True, "message": f"Task '{task['title']}' deleted"})
    except Exception as e:
        print(f"Error deleting task: {e}")
        return jsonify({"success": False, "error": f"Error deleting task: {str(e)}"})

def task_scheduler():
    while True:
        try:
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            tasks_snapshot = db.child("tasks").get()
            tasks = tasks_snapshot.val() if hasattr(tasks_snapshot, 'val') else (tasks_snapshot or {})
            if not tasks:
                tasks = {}
            for task_id, task in tasks.items():
                if task.get("time") == current_time and task.get("status") == "pending":
                    print(f"Reminder: {task['title']} at {task['time']}")
                    db.child("tasks").child(task_id).update({
                        "status": "completed",
                        "completed_at": now.strftime("%Y-%m-%d %H:%M:%S")
                    })
        except Exception as e:
            print(f"Scheduler error: {str(e)}")
        time.sleep(60)

# Time and Stopwatch Routes
@app.route("/get_time", methods=["GET"])
def get_time():
    try:
        current_time = datetime.now().strftime("%I:%M:%S %p")
        return jsonify({"response": f"The current time is {current_time}.", "speak": True})
    except Exception as e:
        print(f"Error in get_time: {str(e)}")
        return jsonify({"response": f"Error fetching time: {str(e)}", "speak": True}), 500

@app.route("/get_date", methods=["GET"])
def get_date():
    try:
        current_date = datetime.now().strftime("%B %d, %Y")
        return jsonify({"response": f"Today's date is {current_date}.", "speak": True})
    except Exception as e:
        print(f"Error in get_date: {str(e)}")
        return jsonify({"response": f"Error fetching date: {str(e)}", "speak": True}), 500

@app.route("/get_tomorrow_date", methods=["GET"])
def get_tomorrow_date():
    try:
        tomorrow_date = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
        return jsonify({"response": f"Tomorrow's date is {tomorrow_date}.", "speak": True})
    except Exception as e:
        print(f"Error in get_tomorrow_date: {str(e)}")
        return jsonify({"response": f"Error fetching tomorrow's date: {str(e)}", "speak": True}), 500

@app.route("/get_yesterday_date", methods=["GET"])
def get_yesterday_date():
    try:
        yesterday_date = (datetime.now() - timedelta(days=1)).strftime("%B %d, %Y")
        return jsonify({"response": f"Yesterday's date is {yesterday_date}.", "speak": True})
    except Exception as e:
        print(f"Error in get_yesterday_date: {str(e)}")
        return jsonify({"response": f"Error fetching yesterday's date: {str(e)}", "speak": True}), 500

@app.route("/get_day", methods=["GET"])
def get_day():
    try:
        current_day = datetime.now().strftime("%A")
        return jsonify({"response": f"Today is {current_day}.", "speak": True})
    except Exception as e:
        print(f"Error in get_day: {str(e)}")
        return jsonify({"response": f"Error fetching day: {str(e)}", "speak": True}), 500

@app.route("/start_stopwatch", methods=["POST"])
def start_stopwatch():
    global stopwatch_running, stopwatch_start_time, stopwatch_elapsed_time
    try:
        if stopwatch_running:
            return jsonify({"response": "Stopwatch is already running.", "speak": True})
        stopwatch_running = True
        stopwatch_start_time = time.time() - stopwatch_elapsed_time
        return jsonify({"response": "Stopwatch started.", "speak": True, "action": "start_stopwatch"})
    except Exception as e:
        print(f"Error in start_stopwatch: {str(e)}")
        return jsonify({"response": f"Error starting stopwatch: {str(e)}", "speak": True}), 500

@app.route("/stop_stopwatch", methods=["POST"])
def stop_stopwatch():
    global stopwatch_running, stopwatch_start_time, stopwatch_elapsed_time
    try:
        if not stopwatch_running:
            return jsonify({"response": "Stopwatch is not running.", "speak": True})
        stopwatch_running = False
        stopwatch_elapsed_time = time.time() - stopwatch_start_time
        elapsed_seconds = int(stopwatch_elapsed_time)
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60
        return jsonify({
            "response": f"Stopwatch stopped. Elapsed time: {minutes} minutes and {seconds} seconds.",
            "speak": True,
            "action": "stop_stopwatch",
            "elapsed_time": stopwatch_elapsed_time
        })
    except Exception as e:
        print(f"Error in stop_stopwatch: {str(e)}")
        return jsonify({"response": f"Error stopping stopwatch: {str(e)}", "speak": True}), 500

@app.route("/reset_stopwatch", methods=["POST"])
def reset_stopwatch():
    global stopwatch_running, stopwatch_start_time, stopwatch_elapsed_time
    try:
        stopwatch_running = False
        stopwatch_start_time = None
        stopwatch_elapsed_time = 0
        return jsonify({"response": "Stopwatch reset.", "speak": True, "action": "reset_stopwatch"})
    except Exception as e:
        print(f"Error in reset_stopwatch: {str(e)}")
        return jsonify({"response": f"Error resetting stopwatch: {str(e)}", "speak": True}), 500

@app.route("/get_stopwatch_time", methods=["GET"])
def get_stopwatch_time():
    global stopwatch_running, stopwatch_start_time, stopwatch_elapsed_time
    try:
        if not stopwatch_running:
            elapsed_time = stopwatch_elapsed_time
        else:
            elapsed_time = time.time() - stopwatch_start_time
        return jsonify({"elapsed_time": elapsed_time})
    except Exception as e:
        print(f"Error in get_stopwatch_time: {str(e)}")
        return jsonify({"error": f"Error fetching stopwatch time: {str(e)}"}), 500

# Enable CORS for all routes
cors = CORS(app, resources={
    r"/chat": {"origins": "*"},
    r"/play_song": {"origins": "*"},
    r"/stop_song": {"origins": "*"},
    r"/detect_sign": {"origins": "*"},
    r"/detect_face": {"origins": "*"},
    r"/register_user": {"origins": "*"},
    r"/get_spotify_token": {"origins": "*"},
    r"/list_songs": {"origins": "*"},
    r"/set_volume": {"origins": "*"},
    r"/get_time": {"origins": "*"},
    r"/get_date": {"origins": "*"},
    r"/get_tomorrow_date": {"origins": "*"},  # Added
    r"/get_yesterday_date": {"origins": "*"},  # Added
    r"/get_day": {"origins": "*"},
    r"/start_stopwatch": {"origins": "*"},
    r"/stop_stopwatch": {"origins": "*"},
    r"/reset_stopwatch": {"origins": "*"},
    r"/get_stopwatch_time": {"origins": "*"},
    r"/add-task": {"origins": "*"},
    r"/get-tasks": {"origins": "*"},
    r"/delete-task": {"origins": "*"},
    r"/get-task-history": {"origins": "*"}
})

# Run Flask App
if __name__ == "__main__":
    scheduler_thread = threading.Thread(target=task_scheduler, daemon=True)
    scheduler_thread.start()
    app.run(debug=True, host='127.0.0.1', port=5000)