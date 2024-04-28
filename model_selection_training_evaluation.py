import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate_model(data):
  
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier()
    }
    
    for model_name, model in models.items():
        print(f"\nTraining and evaluating {model_name}...")
        model.fit(X_train_preprocessed, y_train)
        y_pred = model.predict(X_test_preprocessed)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    
if __name__ == "__main__":

    data = pd.read_csv("heart_disease_dataset.csv")
    
    train_and_evaluate_model(data)
