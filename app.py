from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
import os
import string
import nltk
from rapidfuzz import process

# ── NLTK Data ─────────────────────────────────────────────────────────────────
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')  # Required by newer NLTK versions

# ── App Setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# FIX #2: SECRET_KEY from environment variable — stable across deploys
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-fallback-key-change-in-production')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')  # FIX #1: absolute model path

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ── Database ───────────────────────────────────────────────────────────────────
# FIX #3: Read DATABASE_URL from env (use Render's PostgreSQL in production)
database_url = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(DATA_DIR, 'users.db'))

# SQLAlchemy 1.4+ requires 'postgresql://' not 'postgres://'
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ── User Model ─────────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(256))  # Longer to accommodate scrypt hashes

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── FIX #4: Create DB tables here so Gunicorn (Render) picks it up ─────────────
with app.app_context():
    db.create_all()

# ── FIX #1: Load Models with absolute paths and error handling ─────────────────
try:
    tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    recipes_df = pd.read_pickle(os.path.join(MODEL_DIR, 'recipes_df.pkl'))
    tfidf_matrix = tfidf.transform(recipes_df['soup'])
    cosine_sim = joblib.load(os.path.join(MODEL_DIR, 'cosine_similarity.pkl'))
    indices = pd.Series(recipes_df.index, index=recipes_df['name']).drop_duplicates()
    MODELS_LOADED = True
    print("✅ Models loaded successfully.")
except Exception as e:
    MODELS_LOADED = False
    print(f"⚠️  Could not load models: {e}")
    recipes_df = None
    cosine_sim = None
    indices = None

# ── Text Preprocessing ─────────────────────────────────────────────────────────
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stop_words(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    return ' '.join(word for word in tokens if word.lower() not in stop_words)

def stem_text(text):
    stemmer = nltk.PorterStemmer()
    tokens = nltk.word_tokenize(text)
    return ' '.join(stemmer.stem(word) for word in tokens)

def lemmatize_text(text):
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(word) for word in tokens)

def process_input(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_stop_words(text)
    text = stem_text(text)
    text = lemmatize_text(text)
    return text

def clean_text(text):
    if isinstance(text, list):
        return ', '.join(text)
    text_str = str(text)
    text_str = text_str.replace("] [", ",")
    text_str = text_str.replace("[", "").replace("]", "")
    text_str = text_str.replace("'", "")
    return text_str

def map_nutrition(nutrition_str):
    labels = ["Calories", "Total Fat", "Sugar", "Sodium", "Protein", "Saturated Fat", "Fiber"]
    nutrition_list = [item.strip() for item in nutrition_str.split(",")]
    nutrition_list = (nutrition_list + ["N/A"] * len(labels))[:len(labels)]
    mapped = {}
    for i, value in enumerate(nutrition_list):
        try:
            mapped[labels[i]] = float(value)
        except ValueError:
            mapped[labels[i]] = "N/A"
    return mapped

# ── Recommendation Engine ──────────────────────────────────────────────────────
def get_recommendations(name):
    if not MODELS_LOADED:
        return None

    if name in indices:
        idx = indices[name]
        sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        recipe_indices = [i[0] for i in sim_scores]
    else:
        matches = process.extract(name, recipes_df['name'], limit=10)
        recipe_indices = [indices[match[0]] for match in matches]

    # FIX #7: Use .copy() to avoid SettingWithCopyWarning
    recommendations = recipes_df[
        ['name', 'ingredients', 'minutes', 'steps', 'tags', 'nutrition']
    ].iloc[recipe_indices].copy()

    recommendations['name'] = recommendations['name'].apply(clean_text)
    recommendations['ingredients'] = recommendations['ingredients'].apply(clean_text)
    recommendations['minutes'] = recommendations['minutes'].apply(clean_text)
    recommendations['steps'] = recommendations['steps'].apply(clean_text)
    recommendations['tags'] = recommendations['tags'].apply(clean_text)
    recommendations['nutrition'] = recommendations['nutrition'].apply(clean_text)
    recommendations['nutrition'] = recommendations['nutrition'].apply(map_nutrition)

    return recommendations

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
@app.route('/home')  # keep /home working too
def index():
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
        flash('Login failed. Check your email and password.')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        # Check for duplicate email or username
        if User.query.filter_by(email=email).first():
            flash('An account with that email already exists.')
            return redirect(url_for('signup'))
        if User.query.filter_by(username=username).first():
            flash('That username is already taken.')
            return redirect(url_for('signup'))

        # FIX #6: Removed deprecated method='sha256' — uses secure scrypt by default
        hashed_password = generate_password_hash(password)
        new_user = User(email=email, username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)

        # FIX #5: Redirect to dashboard, not login, after signup
        return redirect(url_for('dashboard'))

    return render_template('signup.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    recommendations = None
    if request.method == 'POST':
        if not MODELS_LOADED:
            flash('Recipe engine is unavailable. Please try again later.')
        else:
            recipe_name = request.form.get('recipe_name')
            recommendations = get_recommendations(recipe_name)
            if recommendations is None or recommendations.empty:
                flash('No similar recipes found. Try a different name.')
                recommendations = None
    return render_template('dashboard.html', name=current_user.username, recommendations=recommendations)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
