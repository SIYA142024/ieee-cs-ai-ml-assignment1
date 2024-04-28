import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):

    print("Summary Statistics:")
    print(data.describe())
    
    print("\nCorrelation Matrix:")
    correlation_matrix = data.corr()
    print(correlation_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=data)
    plt.title("Distribution of Target Variable")
    plt.show()
    
    sns.pairplot(data, hue='target')
    plt.title("Pairplot of Numerical Features")
    plt.show()

if __name__ == "__main__":

    data = pd.read_csv("heart_disease_dataset.csv")
    
    perform_eda(data)
