from data import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    data_directory = "aclimdb/train"
    data, labels = load_data(data_directory)
    print(f"Loaded {len(data)} documents.", flush=True)

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)} samples.", flush=True)
    print(f"Validation set size: {len(X_val)} samples.", flush=True)

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    cls = LogisticRegression()
    cls.fit(X_train_tfidf, y_train)

    X_val_tfidf = vectorizer.transform(X_val)
    val_accuracy = cls.score(X_val_tfidf, y_val)
    print(f"Validation Accuracy: {val_accuracy:.4f}", flush=True)