import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import mlflow
import webbrowser
import tempfile
import os
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report


# Définir le chemin absolu du fichier weatherAUS.csv qui fournit des données météorologiques spécifiques pour certaines de villes d'Australie
chemin_weatherAUS = "C:\\Users\\Utilisateur\\Mon Drive\\PROJET_METEO_DEC23_DS\\PRODUCTION\\Patrick\\data_meteo_patrick\\weatherAUS.csv"

# Charger le fichier weatherAUS.csv qui contient les données des villes d'Australie sous un angle géographique et démographique
weather_data = pd.read_csv(chemin_weatherAUS)

# Afficher les premières lignes du fichier
print(weather_data.head())
print("-"*100)
# Afficher les informations sur le fichier
print(weather_data.info())
print("-"*100)
# Afficher la somme des données manquantes par colonne
print(weather_data.isnull().sum())
print("-"*100)


# Prétraitement des données
# Conversion de la colonne 'Date' en type datetime et extraction des caractéristiques de date
weather_data['Date'] = pd.to_datetime(weather_data['Date'])
weather_data['Year'] = weather_data['Date'].dt.year
weather_data['Month'] = weather_data['Date'].dt.month
weather_data['Day'] = weather_data['Date'].dt.day

# Suppression des colonnes inutiles ou redondantes
weather_data.drop(['Date'], axis=1, inplace=True)

# Remplacement des valeurs 'No' et 'Yes' dans 'RainToday' et 'RainTomorrow'
weather_data['RainToday'] = weather_data['RainToday'].replace({'No': 0, 'Yes': 1}).astype(int)
weather_data['RainTomorrow'] = weather_data['RainTomorrow'].replace({'No': 0, 'Yes': 1}).astype(int)

# Supprimer les lignes où 'RainTomorrow' contient NaN
weather_data.dropna(subset=['RainTomorrow'], inplace=True)

# Séparation des caractéristiques et de la cible, gestion des valeurs NaN dans y
X = weather_data.drop(['RainTomorrow'], axis=1)
y = weather_data['RainTomorrow'].dropna()  # s'assurer qu'il n'y a pas de NaN

# Histogramme des jours de pluie par localisation
fig1 = px.histogram(weather_data, x='Location', title='Localisation vs Jours de Pluie', color='RainToday')
fig1.show()

# Diagramme de répartition de la pluie pour demain
rainTomorrow_distribution = weather_data['RainTomorrow'].value_counts()
fig2 = px.pie(names=rainTomorrow_distribution.index, values=rainTomorrow_distribution.values, title='Répartition de la Pluie pour Demain')
fig2.show()

# Nuage de points Température Minimale vs Température Maximale
fig3 = px.scatter(weather_data.sample(2000), x='MinTemp', y='MaxTemp', color='RainToday', title='Température Minimale vs Température Maximale')
fig3.show()


# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Sélection des caractéristiques numériques et catégorielles
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Pipeline de prétraitement pour les caractéristiques numériques
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Heatmap de la Matrice de Corrélation
corr_matrix = weather_data[numeric_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2g', cmap='viridis')
plt.title('Heatmap de la Matrice de Corrélation')
plt.show()


# Pipeline de prétraitement pour les caractéristiques catégorielles
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Création du pipeline de modélisation
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

# Entraînement du modèle
model.fit(X_train, y_train)

# Évaluation du modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Matrice de Confusion
corr_matrix = weather_data[numeric_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='viridis')
plt.title('Heatmap de la Matrice de Corrélation')
plt.show()

