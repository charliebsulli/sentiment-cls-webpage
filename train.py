import pickle
from data import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model(X, y, max_features=None, ngram_range=(1, 2), max_df=0.8, min_df=1, C=10.0):
    # text tends to a have a long tail of rare words, so we don't need to count everything
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    X_tfidf = vectorizer.fit_transform(X)

    cls = LogisticRegression(C=C)
    cls.fit(X_tfidf, y)
    return cls, vectorizer

if __name__ == "__main__":
    data_directory = "aclimdb/train"
    data, labels = load_data(data_directory)
    print(f"Loaded {len(data)} documents.", flush=True)

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)} samples.", flush=True)
    print(f"Validation set size: {len(X_val)} samples.", flush=True)

    cls, vectorizer = train_model(X_train, y_train)
    print(f"Features: {vectorizer.get_feature_names_out()}", flush=True)
    val_accuracy = cls.score(vectorizer.transform(X_val), y_val)
    print(f"Validation Accuracy: {val_accuracy*100:.2f}", flush=True)

    pickle.dump(cls, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    # check on test set
    test_data_directory = "aclimdb/test"
    test_data, test_labels = load_data(test_data_directory)
    test_accuracy = cls.score(vectorizer.transform(test_data), test_labels)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%", flush=True)