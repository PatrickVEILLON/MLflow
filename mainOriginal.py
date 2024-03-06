from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import mlflow
from sklearn.decomposition import PCA


def create_experiments():
    # Initialisation du client MLflow
    client = mlflow.MlflowClient(tracking_uri="http://127.0.0.1:8888")

    # Définition des noms des expériences
    experiment_names = ['experiment_1.py', 'experiment_2.py']

    # Création des expériences
    for experiment_name in experiment_names:
        client.create_experiment(experiment_name)
        print(f"Expérience '{experiment_name}' créée avec succès.")

# Définir le chemin absolu du fichier weatherAUS.csv qui fournit des données météorologiques spécifiques pour certaines de villes d'Australie
chemin_weatherAUS = "C:\\Users\\Utilisateur\\Mon Drive\\PROJET_METEO_DEC23_DS\\PRODUCTION\\Patrick\\data_meteo_patrick\\weatherAUS.csv"

# Charger le fichier weatherAUS.csv qui contient les données des villes d'Australie sous un angle géographique et démographique
weather_data = pd.read_csv(chemin_weatherAUS)

# Définir le chemin absolu du fichier auNew.csv
chemin_auNew = "C:\\Users\\Utilisateur\\Mon Drive\\PROJET_METEO_DEC23_DS\\PRODUCTION\\Patrick\\data_meteo_patrick\\auNew.csv"

# Charger le fichier auNew.csv
au_data = pd.read_csv(chemin_auNew)

# Renommer la colonne pour une correspondance exacte, dans au_data, les villes sont dans une colonne 'city'
au_data.rename(columns={'city': 'Location'}, inplace=True)

# Fusionner les deux DataFrames
fusion_data = pd.merge(weather_data, au_data, on='Location', how='outer')

print(fusion_data['admin_name'].value_counts())

# Extraire la colonne 'RainTomorrow' et 'RainToday' de weather_data si elle existe
if 'RainTomorrow' in weather_data.columns:
    # Ajouter 'RainTomorrow' à fusion_data si elle n'est pas déjà incluse
    if 'RainTomorrow' not in fusion_data.columns:
        fusion_data['RainTomorrow', 'RainToday'] = weather_data['RainTomorrow', 'RainToday'].astype('Int64')

# Supprimer la colonne 'capital'( pas utile) et 'MinTemp'(pas d'impact sur cible) et 'Evaporation'(pas de variation significative) et country et iso2(faible dépendance) du DataFrame
fusion_data.drop(['capital', 'MinTemp', 'Evaporation', 'country', 'iso2'], axis=1, inplace=True)

# Liste des colonnes pour lesquelles nous voulons vérifier les valeurs NaN
columns_of_interest = [
    'MaxTemp', 'Rainfall', 'Sunshine', 
    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm'
]



# Calcul du pourcentage de valeurs NaN pour les variables d'intérêt
nan_percentages = fusion_data[columns_of_interest].isna().mean() * 100

print('le pourcentage de nan pour les variables d intrêt :', nan_percentages )








# Calculer le pourcentage de valeurs NaN pour chaque colonne
pourcentage_nan = fusion_data.isna().mean() * 100        

# Visualisation de la relation entre "RainTomorrow" et les autres variables
# Diagrammes en barres pour les variables catégorielles
plt.figure(figsize=(12, 6))
num_plots = min(len(fusion_data.select_dtypes(include='object')), 6)  # Nombre maximum de sous-graphiques à créer
for i, column in enumerate(fusion_data.select_dtypes(include='object').columns[:num_plots]):
    plt.subplot(2, 3, i+1)  # Utilisation de num_plots pour déterminer le nombre de sous-graphiques
    sns.countplot(x=column, hue='RainTomorrow', data=fusion_data)
    plt.title(f'{column} vs RainTomorrow')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.legend(title='RainTomorrow', loc='upper right')
plt.tight_layout()
plt.show()



# Test du chi-deux pour les variables catégorielles
from scipy.stats import chi2_contingency
for column in fusion_data.select_dtypes(include='object'):
    contingency_table = pd.crosstab(fusion_data[column], fusion_data['RainTomorrow'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f'Chi-square test p-value for {column}: {p}')


# Supprimer les colonnes non numériques
numeric_data = fusion_data.select_dtypes(include=['float64', 'int64', 'int32'])

# Boîtes à moustaches pour les variables numériques en fonction de la valeur de "RainTomorrow"
plt.figure(figsize=(12, 6))
num_plots = min(len(fusion_data.select_dtypes(include=['float64', 'int64'])), 6)  # Nombre maximum de sous-graphiques à créer
for i, column in enumerate(fusion_data.select_dtypes(include=['float64', 'int64']).columns[:num_plots]):
    plt.subplot(2, 3, i+1)  # Utilisation de num_plots pour déterminer le nombre de sous-graphiques
    sns.boxplot(x='RainTomorrow', y=column, data=fusion_data)
    plt.title(f'{column} vs RainTomorrow')
    plt.xlabel('RainTomorrow')
    plt.ylabel(column)
plt.tight_layout()
plt.show()


# Analyse de variance (ANOVA) pour les variables numériques
from scipy.stats import f_oneway
for column in fusion_data.select_dtypes(include=['float64', 'int64']):
    anova_result = f_oneway(fusion_data[column][fusion_data['RainTomorrow'] == 0], fusion_data[column][fusion_data['RainTomorrow'] == 1])
    print(f'ANOVA p-value for {column}: {anova_result.pvalue}')

# Calculer la matrice de corrélation avec 'RainTomorrow_encoded' incluse
corr_matrix = numeric_data.corr()


# Afficher la matrice de corrélation sous forme de heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.show()

'''
# Remplir les valeurs manquantes dans les colonnes catégorielles suivantes avec le mode
colonnes_categorielles = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
for colonne in colonnes_categorielles:
    mode_value = fusion_data[colonne].mode()[0]  # Le mode peut retourner plusieurs valeurs; on prend la première
    fusion_data[colonne] = fusion_data[colonne].fillna(mode_value)
'''
    

# Encodage One-Hot des villes pour numériser le nom des villes en 0 ou 1 pour ne pas faire de hiérarchie entre les villes
villes_one_hot = pd.get_dummies(fusion_data['Location'], prefix='City')
fusion_data = pd.concat([fusion_data, villes_one_hot], axis=1)

# Encodage One-Hot pour 'RainToday'
fusion_data = pd.get_dummies(fusion_data, columns=['RainToday'])

# Convertir la colonne 'Date' en datetime
fusion_data['Date'] = pd.to_datetime(fusion_data['Date'])

# Extraire l'année, le mois et le jour
fusion_data['Year'] = fusion_data['Date'].dt.year
fusion_data['Month'] = fusion_data['Date'].dt.month
fusion_data['Day'] = fusion_data['Date'].dt.day

# Transformation de la Date en Jours Depuis une Date de Référence
date_reference = fusion_data['Date'].min()
fusion_data['Days_since_ref'] = (fusion_data['Date'] - date_reference).dt.days
fusion_data.drop('Date', axis=1, inplace=True)

# Encodage Label pour les autres colonnes catégorielles
label_encoder = LabelEncoder()
for column in ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainTomorrow', 'country', 'iso2', 'admin_name']:
    fusion_data[column + '_encoded'] = label_encoder.fit_transform(fusion_data[column])
    fusion_data.drop(column, axis=1, inplace=True)

# Identifier les colonnes booléennes
bool_columns = fusion_data.select_dtypes(include='bool').columns.tolist()

# Convertir les colonnes booléennes en entiers
fusion_data[bool_columns] = fusion_data[bool_columns].apply(lambda x: label_encoder.fit_transform(x))

# Convertir les valeures manquantes par des zéros
fusion_data_filled = fusion_data.fillna(0)

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(fusion_data)

# Créer un DataFrame pandas à partir des données standardisées
data_scaled_fusion_data = pd.DataFrame(data_scaled, columns=fusion_data.columns)


'''

# Réduire la dimensionnalité avec PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled_fusion_data)
principal_fusion_data = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Concaténer les composantes principales avec la colonne cible
final_fusion_data = pd.concat([principal_fusion_data, fusion_data['RainTomorrow']], axis=1)

# Afficher la variance expliquée par chaque composante principale
print("Variance expliquée par chaque composante principale:")
print(pca.explained_variance_ratio_)

# Afficher le type de données de chaque colonne
print(fusion_data.dtypes)


# Visualiser les corrélations entre les composantes principales
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='RainTomorrow', data=final_fusion_data)
plt.title('PCA Scatter Plot')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
'''

# Calculer le pourcentage de valeurs NaN pour chaque colonne
pourcentage_nan = fusion_data.isna().mean() * 100

# Afficher le pourcentage de valeurs NaN
print(pourcentage_nan)

print(fusion_data)

fusion_data.info()

print(fusion_data.describe())

# Afficher le type de données de chaque colonne
print(fusion_data.dtypes)

'''
# Calculer la matrice de corrélation
corr_matrix = fusion_data.corr()

# Afficher la matrice de corrélation
print(corr_matrix)

# Heatmap de la matrice de corrélation des variables numériques
plt.figure(figsize=(10, 8))
sns.heatmap(fusion_data.corr(), annot=True, cmap='coolwarm')
plt.show()
'''