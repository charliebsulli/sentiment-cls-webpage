from sklearn.model_selection import train_test_split
from data import load_data
from train import train_model


if __name__ == "__main__":
    data_directory = "aclimdb/train"
    data, labels = load_data(data_directory)
    print(f"Loaded {len(data)} documents.", flush=True)

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    # What can we tune?
    # Feature extraction: max_features, ngram_range, max_df, min_df
    # Model hyperparameters: C

    param_grid = {
        'max_features': [5000, 10000, None],
        'ngram_range': [(1, 1), (1, 2)],
        'max_df': [0.8, 1.0],
        'min_df': [0.05 , 0.1, 1],
        'C': [0.1, 1.0, 10.0]
    }

    results = {}
    for max_features in param_grid['max_features']:
        for max_df in param_grid['max_df']:
            for min_df in param_grid['min_df']:
                for ngram_range in param_grid['ngram_range']:
                    for C in param_grid['C']:                        
                        cls, vectorizer = train_model(X_train, y_train, max_features=max_features, ngram_range=ngram_range, max_df=max_df, min_df=min_df, C=C)
                        
                        val_accuracy = cls.score(vectorizer.transform(X_val), y_val)
                        print(f"Validation Accuracy: {val_accuracy*100:.2f}%", flush=True)

                        results[(max_features, max_df, min_df, ngram_range, C)] = val_accuracy

    print("Grid Search Results:", flush=True)
    for params, accuracy in results.items():
        print(f"Params: max_features={params[0]}, max_df={params[1]}, min_df={params[2]}, ngram_range={params[3]}, C={params[4]} => Validation Accuracy: {accuracy*100:.2f}%", flush=True)