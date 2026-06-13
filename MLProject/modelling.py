import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import dagshub

def main():
    data_path = os.path.join('heart_preprocessed.csv')
    df = pd.read_csv(data_path)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dagshub.init(repo_owner='ridhodiysar', repo_name='Eksperimen_SML_Muhammad-Rasyid-Ridho', mlflow=True)
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    main()