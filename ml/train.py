from preprocessing import load_data, preprocess_data


FILE_PATH = "../data/Bank Customer Churn Prediction.csv"


df = load_data(FILE_PATH)

X_train, X_test, y_train, y_test = preprocess_data(df)

# print("Training Data :", X_train.shape)
# print("Testing Data :", X_test.shape)
# print(X_train.head())
print(y_train.head())