import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import pickle
import argparse
from tqdm import tqdm

def main(folder_path, output_model_path):
    # List to store DataFrames
    dfs = []

    # Loop through each file in the folder
    for filename in tqdm(os.listdir(folder_path), desc='Processing CSV files'):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # Read the CSV file into a DataFrame and append to the list
            df = pd.read_csv(file_path)
            dfs.append(df)

    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(dfs, ignore_index=True)

    # Filter rows containing 'rm' in the 'name' column
    concatenated_df = concatenated_df[concatenated_df['name'].str.contains('rm')]

    # Remove the first five columns from the concatenated DataFrame
    concatenated_df = concatenated_df.iloc[:, 5:]

    # Separate features (X) and target (y) variables
    y = concatenated_df.pop("rel")
    X = concatenated_df

    # Split the data into training and testing sets
    X_train, X_test = X[:6000], X[6000:]
    y_train, y_test = y[:6000], y[6000:]

    # Initialize and train the Gradient Boosting Classifier
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    # Print the accuracy on the training set
    print("Training Accuracy:", clf.score(X_train, y_train))

    # Make predictions on the testing set and print classification report
    preds = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, preds))

    # Save the trained model to a file
    model_filename = os.path.join(output_model_path, 'finalized_model.sav')
    pickle.dump(clf, open(model_filename, 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a Gradient Boosting Classifier.")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing CSV files.")
    parser.add_argument("--output_model_path", type=str, help="Path to save the trained model.")
    args = parser.parse_args()

    main(args.folder_path, args.output_model_path)
