import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):
        self.strategy = strategy
    
    def fit(self, X, y=None):
        if self.strategy == 'most_frequent':
            self.fill = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
        else:
            self.fill = X.mean()
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.fill)

def preprocess_data(data):
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', CustomImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    preprocessed_data = preprocessor.fit_transform(data)
    
    return preprocessed_data

if __name__ == "__main__":
    data = pd.read_csv("heart_disease_dataset.csv")
    
    preprocessed_data = preprocess_data(data)
    
    print("Preprocessed data shape:", preprocessed_data.shape)
