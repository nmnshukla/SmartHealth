import pandas as pd
from sklearn.preprocessing import StandardScaler

def loadandprocess_data():
    df = pd.read_csv('data/heart_disease.csv')
    df.replace('?', pd.NA, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    df['sex'] = df['sex'].map({1: 'male', 0: 'female'})
    df = pd.get_dummies(df, columns=['sex', 'cp', 'restecg', 'slope', 'thal'])
    X = df.drop('target', axis=1)
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler