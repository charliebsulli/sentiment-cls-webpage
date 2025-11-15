# Sentiment Analysis Web Application

**Live Demo:** [Link to Live Demo (Coming Soon!)]

## Project Description
This project is a simple web application that performs sentiment analysis on user-provided text. It features a lightweight Flask backend serving a static HTML frontend. The core of the sentiment analysis is powered by a Logistic Regression model, chosen specifically to optimize for fast API response times, prioritizing speed over the marginal accuracy gains offered by more complex models like BERT.

The model was trained on the IMDb movie review dataset (Large Movie Review Dataset v1.0, circa 2011) from Stanford University. It achieves an accuracy of **89.84%** on the test set.

## Technologies Used
*   **Backend:** Python, Flask
*   **Frontend:** HTML, CSS, JavaScript
*   **Machine Learning:** scikit-learn (Logistic Regression, TfidfVectorizer)
*   **Data:** IMDb Large Movie Review Dataset

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/sentiment-cls-webpage.git
    cd sentiment-cls-webpage
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset (if you wish to retrain the model):**
    The model is pre-trained and included in `model.pkl` and `vectorizer.pkl`. If you want to retrain, download the ACLImdb dataset and place the `aclImdb` folder in the project root.
    [Download link for ACLImdb dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

## Running the Application

1.  **Ensure your virtual environment is active.**

2.  **Run the Flask application:**
    ```bash
    export FLASK_APP=app.py
    flask run
    ```
    (On Windows, use `set FLASK_APP=app.py` instead of `export FLASK_APP=app.py`)

3.  Open your web browser and navigate to `http://127.0.0.1:5000/` to access the sentiment analysis webpage.

## Cross-validation Parameters
*(Placeholder for detailed cross-validation parameters and results if applicable)*

## Project Structure
```
.
├── aclImdb/                # IMDb dataset (if downloaded)
├── templates/
│   └── index.html          # Frontend HTML, CSS, and JavaScript
├── .gitignore
├── app.py                  # Flask application (serves frontend and API)
├── data.py                 # Data loading utilities
├── grid_search.py          # Script for hyperparameter tuning
├── model.pkl               # Pre-trained Logistic Regression model
├── README.md               # Project README
├── requirements.txt        # Python dependencies
├── train.py                # Model training script
└── vectorizer.pkl          # Pre-trained TF-IDF vectorizer
```
