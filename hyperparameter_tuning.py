import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def hyperparameter_tuning(data):
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    rf_classifier = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print("\nBest Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    
    best_rf_classifier = grid_search.best_estimator_
    y_pred_best = best_rf_classifier.predict(X_test)
    
    print("\nEvaluation Metrics (Best Model):")
    print("Accuracy Score:", accuracy_score(y_test, y_pred_best))
    print("Classification Report:")
    print(classification_report(y_test, y_pred_best))

if __name__ == "__main__":
   
    data = pd.read_csv("heart_disease_dataset.csv")
    
    hyperparameter_tuning(data)
