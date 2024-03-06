import matplotlib
from more_itertools import tabulate
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
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tabulate
from Code_direction_objet import TrWindDirection 
from sklearn.impute import KNNImputer


# Initialisation du client MLflow
client = mlflow.MlflowClient(tracking_uri="http://localhost:8888")

# Définition des noms des expériences
experiment_names = ['experiment_1.py', 'experiment_2.py']

# Création des expériences
for experiment_name in experiment_names:
        client.create_experiment(experiment_name)
        print(f"Expérience '{experiment_name}' créée avec succès.")

'''
# Define tracking_uri
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

# Define experiment name, run name and artifact_path name
apple_experiment = mlflow.set_experiment("Apple_Models")
run_name = "first_run"
artifact_path = "rf_apples"        
'''


# Chemin du fichier
chemin_df_49_knn_4_clean_years = "C:\\Users\\Utilisateur\\Mon Drive\\PROJET_METEO_DEC23_DS\\PRODUCTION\\Patrick\\data_meteo_patrick\\df_49_knn_4-clean_years.csv"

# Chargement des données
df = pd.read_csv(chemin_df_49_knn_4_clean_years, parse_dates=['Date'], index_col='Date')


print("Chargement des données - "*5)

# Définir le chemin absolu du fichier weatherAUS.csv qui fournit des données météorologiques spécifiques pour certaines de villes d'Australie
chemin_weatherAUS = "C:\\Users\\Utilisateur\\Mon Drive\\PROJET_METEO_DEC23_DS\\PRODUCTION\\Patrick\\data_meteo_patrick\\weatherAUS.csv"

# Charger le fichier weatherAUS.csv qui contient les données des villes d'Australie sous un angle géographique et démographique
weather_data = pd.read_csv(chemin_weatherAUS)



print("_"*50)
print("Traitement des données - "*5)

# Afficher les premières lignes du fichier
print(weather_data.head())
print("-"*100)

# Afficher les informations sur le fichier
print(weather_data.info())
print("-"*100)

# Afficher la somme des données manquantes par colonne
print(weather_data.isnull().sum())
print("-"*100)

# Calculer le pourcentage de valeurs manquantes par colonne
pourcentage_nan = (weather_data.isnull().sum() / len(weather_data)) * 100

# Créer une DataFrame pour stocker les informations
nan_info = pd.DataFrame({'Nom de la colonne': weather_data.columns,
                         'Pourcentage de NaN': pourcentage_nan})

# Trier les colonnes par pourcentage de NaN décroissant
nan_info = nan_info.sort_values(by='Pourcentage de NaN', ascending=False)

# Afficher la DataFrame avec les informations sur les valeurs manquantes
print("informations dur les valeurs manquantes en pourcentage: \n", nan_info)


# Suppression des données manquantes pour les variables cibles RainToday et RainTomorrow
weather_data.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# Normaliser les données 
scaler = StandardScaler()
weather_data_scaled = scaler.fit_transform(weather_data[['MinTemp', 'MaxTemp']])

# Transformer les données normalisées en DataFrame
weather_data_scaled_df = pd.DataFrame(weather_data_scaled, columns=['MinTemp', 'MaxTemp'])

# Utiliser KNNImputer pour imputer les valeurs manquantes
imputer = KNNImputer(n_neighbors=5) 
weather_data_imputed = imputer.fit_transform(weather_data_scaled_df)

# Transformer les données imputées et re-normalisées en DataFrame
weather_data_imputed_df = pd.DataFrame(scaler.inverse_transform(weather_data_imputed), columns=['MinTemp', 'MaxTemp'])

# Remplacer les colonnes originales par les colonnes imputées
weather_data['MinTemp'] = weather_data_imputed_df['MinTemp']
weather_data['MaxTemp'] = weather_data_imputed_df['MaxTemp']


# Vérifier si la colonne 'RainTomorrow' est présente dans le DataFrame après suppression des données manquantes
if 'RainTomorrow' in weather_data.columns:
    print("La colonne 'RainTomorrow' est toujours présente dans le DataFrame après la suppression des données manquantes.")
else:
    print("La colonne 'RainTomorrow' n'est pas présente dans le DataFrame après la suppression des données manquantes.")

# Afficher la somme des données manquantes par colonne après suppression des lignes
print("Somme des données manquantes : \n" , weather_data.isnull().sum())
print("-"*100)

# Afficher le total des données manquantes de la colonne 'Date'
print('le nombre de NaNs de "Date" est de : ', weather_data['Date'].isnull().sum())
print("-"*100)

# Convertir la colonne 'Date' au format "Datetime"
weather_data['Date'] = pd.to_datetime(weather_data['Date'])

# Réinitialiser l'index de weather_data après le chargement et avant le traitement
weather_data.reset_index(drop=True, inplace=True)

# Instance de la classe pour le traitement des directions du vent
wind_direction_processor = TrWindDirection()

# Traitement des données avec la classe TrWindDirection
weather_data_processed = wind_direction_processor.transform(weather_data)

# Réinitialiser l'index du DataFrame traité
weather_data_processed.reset_index(drop=True, inplace=True)

# Créer une instance de LabelEncoder
label_encoder = LabelEncoder()

# Appliquer l'encodage sur la colonne 'Location'
weather_data['Location'] = label_encoder.fit_transform(weather_data['Location'])

# Maintenant, la colonne 'Location' contient des valeurs numériques représentant les différentes villes.

# Renommer les nouvelles colonnes avec les noms des valeurs catégorielles d'origine
weather_data.rename(columns={'Yes': 'RainTomorrow_Yes'}, inplace=True)


# Vérifier si la colonne 'RainTomorrow' est présente dans le DataFrame après la conversion
if 'RainTomorrow' in weather_data.columns:
    print("La colonne 'RainTomorrow' est toujours présente dans le DataFrame après la conversion.")
else:
    print("La colonne 'RainTomorrow' n'est pas présente dans le DataFrame après la conversion.")


# Convertir la variable RainTomorrow en type numérique
if 'RainTomorrow' in weather_data.columns:
    weather_data['RainTomorrow'] = weather_data['RainTomorrow'].map({'No': 0, 'Yes': 1}).astype(int)
else:
    print("La variable 'RainTomorrow' n'est pas présente dans le DataFrame.")

# Encodage one-hot des variables catégorielles avec exclusion de la colonne Location de l'encodage one-hot
# weather_data_encoded = pd.get_dummies(weather_data, columns=['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow'])

# Encoder les variables catégorielles dans le DataFrame
# weather_data_encoded = encode_categorical_columns(weather_data_encoded)

# Afficher les premières lignes après conversion
print("Premières lignes après conversion : ", weather_data.head())


'''
print("Encodage des variables catégorielles - "*3)


# Tracer une heatmap de la matrice de corrélation encodée
import matplotlib.pyplot as plt
import seaborn as sns



# Calcul de la matrice de corrélation
correlation_matrix_encoded = df.corr()



# Tracer une heatmap de la matrice de corrélation encodée
plt.figure(figsize=(6,6)) # Augmente la taille de la figure
sns.heatmap(correlation_matrix_encoded,  # nom de la variable de la matrice de corrélation
            annot=False, # Désactive l'affichage des valeurs de corrélation
            cmap='coolwarm', # Utilise une palette de couleurs différente
            cbar=True) # Affiche la barre de couleur
plt.title('Matrice de corrélation encodée')
plt.xticks(rotation=90, fontsize=5) # Fait pivoter les étiquettes sur l'axe des x et diminue la taille de la police
plt.yticks(rotation=0, fontsize=5) # Fait pivoter les étiquettes sur l'axe des y et diminue la taille de la police
plt.show()



print(" Analyse de la corrélation par groupement - "*3) 
# Backup du dataset d'origine
weather_data_backup = df.copy()
# Créer un nouveau DataFrame sans la colonne 'index'
weather_data_backup = df.drop(0, axis=0)


# Afficher les colonnnes du fichier
print("Liste des colonnes du fichier 'weather_data_backup' : \n", weather_data_backup.head(), df.columns)


# Groupement des données par la colonne 'Location' et calcul des statistiques descriptives
grouped_data = df.groupby('Location').agg('mean')

# Calcul de la matrice de corrélation
correlation_matrix_grouped = grouped_data.corr()

# Importer les modules nécessaires
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io

# Tracer une heatmap de la matrice de corrélation par groupement
plt.figure(figsize=(6,6)) # Augmente la taille de la figure
sns.heatmap(correlation_matrix_grouped,  # nom de la variable de la matrice de corrélation
            annot=False, # Désactive l'affichage des valeurs de corrélation
            cmap='coolwarm', # Utilise une palette de couleurs différente
            cbar=True) # Affiche la barre de couleur
plt.title('Matrice de Corrélation par Groupement')
plt.xticks(rotation=90, fontsize=5) # Fait pivoter les étiquettes sur l'axe des x et diminue la taille de la police
plt.yticks(rotation=0, fontsize=5) # Fait pivoter les étiquettes sur l'axe des y et diminue la taille de la police
plt.show()


print(" Utilisation de techniques d'analyse multivariée - " *3)
# Backup du dataset d'origine
weather_data_backup = df.copy()
# Créer un nouveau DataFrame sans la colonne 'index'
weather_data_backup = df.drop(0, axis=0)

# Analyse multivariée avec MCA
from prince import MCA

# Vérification des types de données
print(df.info())

# Gestion des valeurs manquantes
print(df.isnull().sum())

# Imputation des valeurs manquantes 
# Remplacez NaN par la moyenne dans toutes les colonnes numériques
df.fillna(df.mean(), inplace=True)

# Supprimer la colonne 'Date'
df.drop(columns=['Date'], inplace=True)

import matplotlib.pyplot as plt
from prince import MCA
import pandas as pd

# Exécution de l'analyse multivariée avec MCA sur le DataFrame encodé
mca_encoded = MCA(n_components=2)
mca_encoded.fit(df)
mca_coordinates_encoded = mca_encoded.transform(df)

# Exécution de l'analyse multivariée avec MCA sur le DataFrame original
mca = MCA(n_components=2)
mca.fit(df)
mca_coordinates = mca.transform(weather_data)

# Création d'une grille de graphiques avec 1 ligne et 2 colonnes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot pour weather_data_encoded
scatter1 = ax1.scatter(mca_coordinates_encoded[0], mca_coordinates_encoded[1], c=df['RainTomorrow_1'])
ax1.set_title('MCA Components with weather_data_encoded')
ax1.set_xlabel('First principal component')
ax1.set_ylabel('Second principal component')
fig.colorbar(scatter1, ax=ax1, label='RainTomorrow')

# Scatter plot pour weather_data
scatter2 = ax2.scatter(mca_coordinates[0], mca_coordinates[1], c=weather_data['RainTomorrow'])
ax2.set_title('MCA Components with weather_data')
ax2.set_xlabel('First principal component')
ax2.set_ylabel('Second principal component')
fig.colorbar(scatter2, ax=ax2, label='RainTomorrow')

plt.show()


print("Matrice de corrélation pour les variables numériques - " *5)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Backup du dataset d'origine
weather_data_backup = df.copy()
# Créer un nouveau DataFrame sans la colonne 'index'
weather_data_backup = df.drop(0, axis=0)


# Calculer la matrice de corrélation
correlation_matrix = df.corr()

# Tracer une heatmap de la matrice de corrélation 
plt.figure(figsize=(6,6)) # Augmente la taille de la figure
sns.heatmap(correlation_matrix,  # nom de la variable de la matrice de corrélation
            annot=False, # Désactive l'affichage des valeurs de corrélation
            cmap='coolwarm', # Utilise une palette de couleurs différente
            cbar=True) # Affiche la barre de couleur
plt.title('Matrice de Corrélation pour les variables numériques')
plt.xticks(rotation=90, fontsize=5) # Fait pivoter les étiquettes sur l'axe des x et diminue la taille de la police
plt.yticks(rotation=0, fontsize=5) # Fait pivoter les étiquettes sur l'axe des y et diminue la taille de la police
plt.show()


print("Analyse de la corrélation bivariée - "*5)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Supprimer les lignes où RainTomorrow est NaN pour assurer la qualité des données
df.dropna(subset=['RainTomorrow'], inplace=True)

# Encoder RainTomorrow en format numérique pour faciliter les calculs
# 'Yes' pour la pluie devient 1 et 'No' pour l'absence de pluie devient 0
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

# Grouper les données par 'Location' et calculer la proportion de jours avec pluie
proportions_pluie_par_location = df.groupby('Location')['RainTomorrow'].mean().reset_index()

# Trier les données pour améliorer la visualisation
proportions_pluie_par_location = proportions_pluie_par_location.sort_values(by='RainTomorrow', ascending=False)

# Visualisation des proportions de jours pluvieux par localisation
plt.figure(figsize=(12, 8))
sns.barplot(x='RainTomorrow', y='Location', data=proportions_pluie_par_location)
plt.xlabel('proportions de jours pluvieux par localisation')
plt.ylabel('Location')
plt.title('Proportion of Rain Days Tomorrow by Location')
plt.xticks(rotation=45)
plt.tight_layout()  # Ajuste la mise en page pour éviter le chevauchement
plt.show()

# Objectif de visualisation: La visualisation vise à montrer la relation entre la direction du vent lors des rafales 
# et la probabilité de pluie le lendemain. Ici, l'accent est mis sur comment la direction du vent affecte la probabilité 
# de pluie, plutôt que sur une proportion précise par catégorie.
# WindGustDir est une variable catégorielle des rafales et RainTomorrow encodée comme 0 ou 1
plt.figure(figsize=(12, 8))
sns.barplot(x="WindGustDir", y="RainTomorrow", data=weather_data, estimator=lambda x: sum(x==1)*100.0/len(x))
plt.ylabel('Percentage of Rain Tomorrow')
plt.title('Relation Between Wind Gust Direction and Rain Tomorrow')
plt.xticks(rotation=45)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Liste des variables numériques et catégorielles à comparer
num_variables = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
                 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']

# Inclure aussi la variable cible non encodée pour les graphiques
df['RainTomorrow'] = weather_data['RainTomorrow'].copy()

# Parcourir chaque variable catégorielle ou numérique et créer des graphiques côte à côte
for variable in num_variables:
    # Initialiser une grille de graphiques avec 1 ligne et 2 colonnes
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Premier graphique avec weather_data
    sns.boxplot(x='RainTomorrow', y=variable, data=weather_data, ax=ax[0])
    ax[0].set_title(f'Original: {variable} vs RainTomorrow')
    
    # Deuxième graphique avec weather_data_encoded
    sns.boxplot(x='RainTomorrow', y=variable, data=df, ax=ax[1])
    ax[1].set_title(f'Encoded: {variable} vs RainTomorrow')
    
    plt.tight_layout()  # Ajuste la mise en page pour éviter le chevauchement
    plt.show()

'''

