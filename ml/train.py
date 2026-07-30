from preprocessing import load_data, preprocess_data


FILE_PATH = "../data/Bank Customer Churn Prediction.csv"


df = load_data(FILE_PATH)

# X_train, X_test, y_train, y_test = preprocess_data(df)
(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
) = preprocess_data(df)

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print(X_train_processed)

# print(X_test_processed.shape)
