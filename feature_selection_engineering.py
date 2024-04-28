import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures

def select_and_engineer_features(data):

    X = data.drop('target', axis=1)
    y = data['target']
    selector = SelectKBest(score_func=f_classif, k=5)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print("Selected Features:")
    print(selected_features)
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    poly_feature_names = poly.get_feature_names(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)
    print("\nPolynomial Features:")
    print(X_poly_df.head())
    
    return X_selected, X_poly_df

if __name__ == "__main__":

    data = pd.read_csv("heart_disease_dataset.csv")
    
    X_selected, X_poly_df = select_and_engineer_features(data)
