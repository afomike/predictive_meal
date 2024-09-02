from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
import os
import string
import nltk
from rapidfuzz import process  # Import RapidFuzz for fuzzy matching

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Flask app and configurations
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()

# Set up database
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created directory: {DATA_DIR}")

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(DATA_DIR, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load the saved models and data
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
recipes_df = pd.read_pickle('models/recipes_df.pkl')

# Transform the 'soup' column to create TF-IDF matrix
tfidf_matrix = tfidf.transform(recipes_df['soup'])

# Load the precomputed cosine similarity matrix
cosine_sim = joblib.load('models/cosine_similarity.pkl')

# Create a reverse map of indices and recipe names
indices = pd.Series(recipes_df.index, index=recipes_df['name']).drop_duplicates()

# Preprocessing functions
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stop_words(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def stem_text(text):
    stemmer = nltk.PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def lemmatize_text(text):
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

def process_input(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_stop_words(text)
    text = stem_text(text)
    text = lemmatize_text(text)
    return text

def clean_text(text):
    # Check if the text is a list
    if isinstance(text, list):
        # Join list elements with a comma and space, and return as a string
        cleaned_text = ', '.join(text)
    else:
        # If not a list, handle it as a string
        text_str = str(text)
        # Remove '] [' by replacing them with ','
        cleaned_text = text_str.replace("] [", ",")
        # Remove the outer brackets if needed
        cleaned_text = cleaned_text.replace("[", "").replace("]", "")
                
        # Remove the outer brackets if needed
        cleaned_text = cleaned_text.replace("'", "")
    return cleaned_text


def get_recommendations(name):
    if name in indices:
        # Exact match: get recommendations based on precomputed cosine similarity
        idx = indices[name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Exclude the recipe itself
        recipe_indices = [i[0] for i in sim_scores]
        recommendations = recipes_df[['name', 'ingredients', 'minutes', 'steps', 'tags']].iloc[recipe_indices]
    else:
        # Use RapidFuzz to find the closest matching recipe names
        matches = process.extract(name, recipes_df['name'], limit=10)
        matched_indices = [indices[match[0]] for match in matches]
        recommendations = recipes_df[['name', 'ingredients', 'minutes', 'steps', 'tags']].iloc[matched_indices]

    # Clean the relevant fields
    recommendations['name'] = recommendations['name'].apply(clean_text)
    recommendations['ingredients'] = recommendations['ingredients'].apply(clean_text)
    recommendations['minutes'] = recommendations['minutes'].apply(clean_text)
    recommendations['steps'] = recommendations['steps'].apply(clean_text)
    recommendations['tags'] = recommendations['tags'].apply(clean_text)

    return recommendations


# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check your email and password.')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(email=email, username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    recommendations = None
    if request.method == 'POST':
        recipe_name = request.form.get('recipe_name')
        recommendations = get_recommendations(recipe_name)
        if recommendations is None or recommendations.empty:
            flash('No similar recipes found.')
            recommendations = None  # Ensure it's None if empty or None
    return render_template('dashboard.html', name=current_user.username, recommendations=recommendations)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
