from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd


# Treinando o modelo
df = pd.read_excel("data/iris data.xls")

def treinar_modelo(model_class, class1, class2):
    df_filtered = df[df['Species'].isin([class1, class2])]
    X = df_filtered.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(df_filtered['Species'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = model_class()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred