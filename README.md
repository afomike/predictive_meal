# Flavr — Smart Meal Recommendations

> Type any dish or ingredient — our recommendation engine finds the perfect recipes tailored to your taste, matched by nutrition, time, and flavour profile.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://predictive-meal.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](./LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Locally](#running-locally)
- [Usage](#usage)
- [Deployment](#deployment)
- [Model Artifacts](#model-artifacts)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Flavr** is a Flask-based meal recommendation web application that combines content-based machine learning with natural language processing to deliver personalised recipe suggestions in under a second. Users create an account, enter any dish they are craving or curious about, and Flavr surfaces the closest matching recipes — complete with full ingredient lists, step-by-step cooking instructions, and nutritional breakdowns.

The recommendation engine is built on TF-IDF vectorisation and cosine similarity, with fuzzy matching via RapidFuzz to handle partial or imprecise recipe queries gracefully.

---

## Live Demo

**[https://predictive-meal.onrender.com/](https://predictive-meal.onrender.com/)**

> **Note:** The demo is hosted on Render's free tier. The first request after a period of inactivity may take 30–60 seconds as the instance wakes from a cold start.

---

## Features

| Feature | Description |
|---|---|
| **AI Recipe Matching** | Content-based filtering using TF-IDF and cosine similarity |
| **Fuzzy Search** | Handles partial or misspelled recipe names via RapidFuzz |
| **Full Nutrition Data** | Every recommendation includes calories, macros, and dietary flags |
| **Step-by-Step Instructions** | Structured cooking guides rendered with each result |
| **User Authentication** | Secure sign-up, login, and protected dashboard using Flask-Login |
| **Configurable Database** | Defaults to SQLite locally; supports any SQLAlchemy-compatible URI in production |

---

## How It Works

```
01  👤  Create your account   →  Sign up in under 30 seconds — no credit card required
02  🔍  Enter a recipe name   →  Type any dish you love or want to explore
03  🤖  AI finds matches      →  The model analyses ingredients, prep time, and nutrition to rank results
04  🍽️  Cook with confidence  →  Get full ingredients, instructions, and nutrition info instantly
```

---

## Architecture

```
User Browser
     │
     ▼
Flask Application (app.py)
     │
     ├── /                →  Landing page
     ├── /signup          →  User registration
     ├── /login           →  User authentication
     └── /dashboard       →  Recipe search & recommendation results
           │
           ▼
     Recommendation Engine
     ├── tfidf_vectorizer.pkl    (TF-IDF vectorizer fitted on recipe corpus)
     ├── recipes_df.pkl          (serialized recipe dataset)
     └── cosine_similarity.pkl   (precomputed similarity matrix)
           │
           ▼
     Database (SQLite / configurable via DATABASE_URL)
     └── users.db                (user accounts and sessions)
```

---

## Project Structure

```
flavr/
│
├── app.py                      # Flask application, routes, and recommendation engine
├── requirements.txt            # Pinned Python dependencies
├── predictive_meal.ipynb       # Notebook for data exploration and model development
│
├── models/
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
│   ├── recipes_df.pkl          # Serialized recipe DataFrame
│   └── cosine_similarity.pkl   # Precomputed cosine similarity matrix
│
├── data/
│   └── users.db                # SQLite database (auto-created on first run)
│
├── templates/                  # Jinja2 HTML templates
│   ├── index.html              # Landing page
│   ├── signup.html             # Registration form
│   ├── login.html              # Login form
│   └── dashboard.html          # Recipe search and results view
│
├── static/                     # Static assets (CSS, images)
│
├── README.md
└── LICENSE
```

---

## Prerequisites

- **Python 3.11 or higher**
- `conda` or `venv` for environment isolation
- All model artifacts present in `models/` (see [Model Artifacts](#model-artifacts))

---

## Installation

**1. Clone the repository.**

```bash
git clone https://github.com/your-username/flavr.git
cd flavr
```

**2. Create and activate a Python environment.**

Using `conda`:
```bash
conda create -n flavr python=3.11 -y
conda activate flavr
```

Using `venv`:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies.**

```bash
pip install -r requirements.txt
```

**4. Confirm model artifacts are in place.**

```bash
ls models/
# Expected: tfidf_vectorizer.pkl  recipes_df.pkl  cosine_similarity.pkl
```

---

## Configuration

Flavr reads the following environment variables at startup:

| Variable | Description | Default |
|---|---|---|
| `SECRET_KEY` | Flask session signing secret | A development fallback (insecure for production) |
| `DATABASE_URL` | SQLAlchemy-compatible database URI | `sqlite:///data/users.db` |

Set these in a `.env` file or your deployment environment before starting the application in production.

```bash
# Example .env
SECRET_KEY=your-very-secure-random-secret
DATABASE_URL=postgresql://user:password@host:5432/flavr
```

---

## Running Locally

```bash
python app.py
```

Then open your browser and navigate to:

| Route | Description |
|---|---|
| `http://127.0.0.1:5000/` | Landing page |
| `http://127.0.0.1:5000/signup` | Create a new account |
| `http://127.0.0.1:5000/login` | Log in to your account |
| `http://127.0.0.1:5000/dashboard` | Recipe search and recommendations (requires login) |

The SQLite database at `data/users.db` is created automatically on first run if it does not already exist.

---

## Usage

1. Navigate to the landing page and click **Get Started**.
2. Complete the sign-up form to create an account.
3. Log in with your credentials to access the dashboard.
4. Enter any recipe name or dish in the search box.
5. Flavr returns the closest matching recipes with full nutrition data and cooking instructions.

---

## Deployment

For production, serve the application with a WSGI server such as **Gunicorn**:

```bash
gunicorn --bind 0.0.0.0:8000 app:app
```

### Deploying to Render

1. Connect your GitHub repository to a new Render **Web Service**.
2. Set the **Build Command** to `pip install -r requirements.txt`.
3. Set the **Start Command** to `gunicorn app:app`.
4. Add `SECRET_KEY` and `DATABASE_URL` as **Environment Variables** in the Render dashboard.
5. Ensure the `models/` directory and its `.pkl` artifacts are committed to the repository.

---

## Model Artifacts

The recommendation engine relies on three serialized artifacts in `models/`:

| File | Purpose |
|---|---|
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer fitted on the recipe text corpus |
| `recipes_df.pkl` | Serialized pandas DataFrame containing the full recipe dataset |
| `cosine_similarity.pkl` | Precomputed cosine similarity matrix for fast nearest-neighbour lookup |

These artifacts are generated by the `predictive_meal.ipynb` notebook. If any artifact is missing or incompatible with the installed version of `scikit-learn`, the application will raise an error at startup.

---

## Dependencies

Core dependencies (all versions pinned in `requirements.txt`):

| Package | Role |
|---|---|
| `Flask` | Web framework and routing |
| `Flask-Login` | User session and authentication management |
| `Flask-SQLAlchemy` | ORM and database integration |
| `pandas` | Recipe data handling and DataFrame operations |
| `scikit-learn` | TF-IDF vectorisation and cosine similarity |
| `rapidfuzz` | Fuzzy matching for tolerant recipe name search |
| `joblib` | Model artifact serialization and loading |
| `nltk` | Natural language preprocessing |
| `Werkzeug` | Password hashing and security utilities |

---

## Contributing

Contributions, suggestions, and bug reports are welcome. If you add new features or change the application structure, please:

1. Fork the repository and create a feature branch.
2. Update this README with any new setup steps, routes, or required artifacts.
3. Open a pull request with a clear description of the change.

---

## License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE) for the full terms.
