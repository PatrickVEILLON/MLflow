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
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt


'''
def create_experiments():
    # Initialisation du client MLflow
    client = mlflow.MlflowClient(tracking_uri="http://127.0.0.1:8888")

    # Définition des noms des expériences
    experiment_names = ['experiment_1.py', 'experiment_2.py']

    # Création des expériences
    for experiment_name in experiment_names:
        client.create_experiment(experiment_name)
        print(f"Expérience '{experiment_name}' créée avec succès.")
'''

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
# Suppression des données manquantes pour les variables cibles RainToday et RainTomorrow
weather_data.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# Afficher la somme des données manquantes par colonne
print(weather_data.isnull().sum())
print("-"*100)
# Afficher le total des données manquante de la collonne 'Date'
print('le nombre de NaNs de "Date" est de : ', weather_data['Date'].isnull().sum())
print("-"*100)
# Convertir la colonne 'Date' au format "Datetime"
weather_data['Date']=pd.to_datetime(weather_data['Date'])

# Prétraitement spécifique pour 'Evaporation'
def convert_evaporation_range(value):
    if '-' in str(value):
        low, high = map(float, value.split('-'))
        return (low + high) / 2
    try:
        return float(value)
    except ValueError:
        return np.nan

weather_data['Evaporation'] = weather_data['Evaporation'].apply(convert_evaporation_range)
weather_data['Evaporation'].fillna(weather_data['Evaporation'].mean(), inplace=True)


# Séparation des colonnes de données numériques des colonnes des données catégorielles pour traitement séparé
numeric_col=weather_data.select_dtypes(include=np.number).columns.tolist()
category_col=weather_data.select_dtypes(include='object').columns.tolist()

print('liste des colonnes numériques: ', numeric_col)
print("-"*100)
print('liste des colonnes catégorielles : ',category_col)
print("-"*100)

# Traitement des collonnes catégorielles
# Afficher la somme de données manquantes pour liste des colonnes catégorielles
print('liste des données manquantes pour la liste des colones catégorielles : \n', weather_data[category_col].isnull().sum()) 
print("-"*100)

# Affichage des valeurs de chaque colonne catégorielle
for col in weather_data[category_col]:
    print(col)
    print("-"*100)
    print(weather_data[col].value_counts())
print("-"*100)

# Affichage des données manquantes des variables cibles pour vérification
print(weather_data['RainToday'].isnull().sum())
print(weather_data['RainTomorrow'].isnull().sum())
print("-"*100)

# Remplacer les données manquantes des colonnes catégorielles par 'unknown' pour la suite du traitement
weather_data[category_col] = weather_data[category_col].fillna('Unknown')

# Affichage de la somme des données manquantes pour les colonnes catégorielles
print('Nombre de données manquantes dans les colonnes catégorielles : \n' , weather_data[category_col].isnull().sum())
print("-"*100)

# Traitement des collonnes numériques
# Afficher les premières lignes des colonnes numériques
print('Premières lignes des colonnes numériques : \n', weather_data[numeric_col].head())
print("-"*100)

# Affichage de la somme des données manquantes poiur les colonnes numériques
print('Somme des données manquantes pour les colonnes nuémriques : \n' ,weather_data[numeric_col].isnull().sum())
print()
# Calcul du pourcentage de valeurs NaN pour les colonnes numériques
nan_percentages_num_col = weather_data[numeric_col].isna().mean() * 100
print('le pourcentage de nan pour les colonnes nuémriques : \n', nan_percentages_num_col )
print("-"*100)
# Affichage de la description statistique des colonnes numériques
print('Tableau des données statistiques des colonnes numériques : \n',  weather_data[numeric_col].describe().T)
print("-"*100)

# Traitement des valeurs manquantes dans les colonnes numériques par la moyennes des valeurs non manquantes
imputer=SimpleImputer(strategy='mean')
imputer.fit(weather_data[numeric_col])
# Imputer les valeurs manquantes
weather_data[numeric_col] = imputer.transform(weather_data[numeric_col])
# Afficher les premières lignes des colonnes numériques
print('Premières lignes des colonnes numériques : \n', weather_data[numeric_col].head())
print("-"*100)

# Affichier les statistiques de la moyenne de chaque colonne numérique
imputer_stats=list(imputer.statistics_)
print('Statistic des colonnes : \n', imputer_stats)
print("-"*100)

# Remplacer les données manquantes des colonnes numériques par la moyenne calculée
transformed_data = imputer.transform(weather_data[numeric_col])
print('Tableau NumPy des données transformées : \n', transformed_data)
print("-"*100)

# Vérification que le dataset n'a plus de données manquantes
# Affichage des données manquantes des colonnes numériques
print('liste de la somme des données manquantes des colonnes numériques : \n' , weather_data[numeric_col].isnull().sum())
print("-"*100)
# Affichage des données manquantes des colonnes catégorielles
print('liste de la somme des données manquantes des colonnes catégorielles : \n' , weather_data[category_col].isnull().sum())
print("-"*100)
print("-"*100)

# Analyse exploratoire
# Configuration du style esthétique des graphiques avec seaborn
sns.set_style('dark')
# Configuration de la taille de la police pour les graphiques
matplotlib.rcParams['font.size'] = 10
# Configuration des dimensions par défaut des figures
matplotlib.rcParams['figure.figsize'] = (8, 4)
# Configuration de la couleur de fond de la figure
matplotlib.rcParams['figure.facecolor'] = 'White'
# Affichage de l'histogramme 
fig1 = px.histogram(weather_data,x='Location', title=' fig1 - Location Vs rainDays', color='RainToday')
fig1.show()
print("--Figure 1 histogramme des jours de pluie--"*3)
print("-"*100)

# Affichage du graphique de corrélation entre les jours de pluie de RainToday et RainTomorrow
rainTomorrowNbreDays = weather_data['RainTomorrow'].value_counts()
# Création du graphique en barres avec Plotly Express
fig2 = px.bar(data_frame=rainTomorrowNbreDays, x=rainTomorrowNbreDays.index, y=rainTomorrowNbreDays.values )
# Ajout des étiquettes
fig2.update_layout(
    title='fig2 - Répartition des jours avec pluie demain (RainTomorrow)', # Titre du graphique
    xaxis_title='Pluie demain', # Titre de l'axe des abscisses
    yaxis_title='Nombre de jours', # Titre de l'axe des ordonnées
    legend_title='Pluie le jour suivant' # Titre de la légende
)
# Affichage du graphique
fig2.show()

# Impression des informations sur le graphique
print("--Figure 2 histogramme de comparaison des jours de pluie RainToday et RainTomorrow --"*2)
print("-"*100)

# Calcul de la distribution de la variable 'RainTomorrow'
rainTomorrowDistribution = weather_data['RainTomorrow'].value_counts()

# Création du diagramme en camembert avec Plotly Express
fig3 = px.pie(
    names=rainTomorrowDistribution.index, # Les catégories pour les segments du camembert
    values=rainTomorrowDistribution.values, # Les valeurs pour chaque segment
    title='fig3 - Répartition de la pluie pour demain (RainTomorrow)', # Titre du graphique
    color_discrete_sequence=px.colors.sequential.RdBu # Séquence de couleurs pour les segments
)
# Personnalisation supplémentaire pour améliorer l'affichage
# Afficher le pourcentage et le label sur le graphique, et tirer légèrement le segment le plus grand
fig3.update_traces(textinfo='percent+label', pull=[0.1, 0]) 
# Ajustement de la légende et de ses marges
fig3.update_layout(
    legend_title_text='Prévision de pluie', # Titre de la légende
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02, # Positionne la légende au-dessus du diagramme en camembert
        xanchor="center",
        x=0.5
    ),
    margin=dict(t=50, b=10, l=10, r=10), # Marges autour du graphique
)
# Afficher le diagramme en camembert
fig3.show()

print("-"*100)
print("Backup WeatherAUS_Bachuped.csv"*5)
print("-"*100)
# Créer un fichier de restauration du fichier de données sans données manquantes
weather_data.to_csv('WeatherAUS_Bachuped.csv')
backup=pd.DataFrame(weather_data)
print('Affichage des premières données du fichier weather_data_Backup.csv : \n' , backup.head())
print('Affichage de 10 lignes aléatoire du fichier weather_data_Backup.csv : \n' ,backup.sample(10))
print("-"*100)
print("-"*100)

# Visualisation de la relation entre "RainTomorrow" et les autres variables
# Diagrammes en barres pour les variables catégorielles
fig4 = plt.figure(figsize=(12, 6))
num_plots = min(len(category_col), 6)  # Nombre maximum de sous-graphiques à créer
for i, column in enumerate(category_col[:num_plots]):
    plt.subplot(2, 3, i+1)  # Utilisation de num_plots pour déterminer le nombre de sous-graphiques
    sns.countplot(x=column, hue='RainTomorrow', data=weather_data)
    plt.title(f'{column} fig4 - Allcategories VS RainTomorrow')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.legend(title='RainTomorrow', loc='upper right')
# Afficher les graphiques dans le navigateur
fig4.show()  

# Test du chi-deux pour les variables catégorielles
for column in category_col:
    contingency_table = pd.crosstab(weather_data[column], weather_data['RainTomorrow'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f'Chi-square test p-value for {column}: {p}')
# Chi-square test p-value for [Variable]: 0.0 : Une valeur-p de 0.0 (ou une valeur très proche de zéro) pour le test du chi-carré indique que, 
# statistiquement, il y a une association très forte entre la variable en question et la variable de réponse (par exemple, la pluie le lendemain). 
# Une valeur-p très faible (typiquement moins de 0.05) signifie que l'on rejette l'hypothèse nulle d'indépendance, ce qui suggère que les deux variables 
# testées sont probablement dépendantes l'une de l'autre.
# Pour les variables Location, WindGustDir, WindDir9am, RainToday, et RainTomorrow, les valeurs-p de 0.0 suggèrent qu'il existe une relation significative 
# entre ces variables et la variable de résultat (probablement la pluie le lendemain). Pour la variable WindDir3pm, la valeur-p est extrêmement faible, 
# mais pas exactement zéro, ce qui indique également une relation très forte, bien que légèrement moins prononcée que pour les autres variables.

# Création de l'histogramme avec Plotly Express
fig5 = px.histogram(
    weather_data, 
    x='MinTemp', 
    title='fig5 - Température minimale avec RainTomorrow', 
    color='RainTomorrow'
)
# Affichage de l'histogramme
fig5.show()

# Création de l'histogramme avec Plotly Express
fig6 = px.histogram(
    weather_data, 
    x='WindDir9am', 
    title='fig6 - Direction du vent à 9h avec RainTomorrow', 
    color='RainTomorrow'
)
# Affichage de l'histogramme
fig6.show()
# D'après le schéma, il ne semble pas y avoir une corrélation forte et directe entre la direction du vent le matin (WindDir9am) 
# et la pluie le lendemain (RainTomorrow) simplement en observant les hauteurs des barres. Le graphique montre la distribution de la direction du vent 
# lorsque qu'il a plu le lendemain (en rouge) par rapport à quand il n'a pas plu (en bleu). Bien que certaines directions du vent semblent avoir des 
# proportions légèrement plus élevées de pluie le lendemain (par exemple ESE et NNW), il n'y a pas de tendance clairement définie qui indiquerait une corrélation forte.

# Création de l'histogramme avec Plotly Express
fig7 = px.histogram(
    weather_data, 
    x='WindGustDir', 
    title='fig7 - Direction des rafales de vent avec RainTomorrow', 
    color='WindGustDir'
)
# Affichage de l'histogramme
fig7.show()
# Ce graphique montre le nombre d'occurrences (count) de la température minimale enregistrée avec la variable 'WindGustDir', 
# qui représente la direction des rafales de vent. Ce graphique ne semble pas indiquer une corrélation 
# forte entre une direction particulière de la rafale de vent et la probabilité de pluie le lendemain. Les hauteurs des barres varient 
# avec la direction du vent, mais aucune direction ne se démarque de manière significative par rapport aux autres.

'''
# Création du nuage de points avec Plotly Express
fig8 = px.scatter(
    weather_data.sample(2000),  # Utilisation d'un échantillon de 2000 lignes pour des raisons de visualisation
    title='fig8 - Température minimale vs Température maximale',
    x='MinTemp',
    y='MaxTemp',
    color='RainToday'
)
# Affichage du nuage de points
fig8.show()

'''

# Création du nuage de points avec Plotly Express
fig9 = px.scatter(
    weather_data.sample(2000),  # Utilisation d'un échantillon de 2000 lignes pour des raisons de visualisation
    title='fig9 - Évaporation vs Température minimale',
    x='Location',
    y='Evaporation',
    color='RainToday'
)
# Affichage du nuage de points
fig9.show()


# Création du nuage de points avec Plotly Express
fig10 = px.scatter(
    weather_data.sample(2000),  # Utilisation d'un échantillon de 2000 lignes pour des raisons de visualisation
    x='Evaporation',
    title='fig10 - Évaporation avec RainTomorrow',
    color='RainTomorrow'
)
# Affichage du nuage de points
fig10.show()

print("-"*100)

# Définition des bins pour les plages d'évaporation
bins = [0, 2, 4, 6, 8, 10]
# Ajout d'une nouvelle colonne 'EvapRange' pour catégoriser les valeurs d'évaporation
weather_data['EvapRange'] = pd.cut(weather_data['Evaporation'], bins=bins, labels=['0-2', '2-4', '4-6', '6-8', '8-10'])

# Création d'une table de contingence pour compter les occurrences
cross_tab = pd.crosstab(weather_data['EvapRange'], weather_data['RainTomorrow'])

# Utilisation de Plotly pour créer un diagramme à barres empilé
fig11 = go.Figure()
for col in cross_tab.columns:
    fig11.add_trace(go.Bar(x=cross_tab.index, y=cross_tab[col], name=str(col)))

# Mise à jour du layout pour empiler les barres
fig11.update_layout(barmode='stack', title_text='fig11 - Évaporation vs RainTomorrow',
                  xaxis_title='Plage d\'évaporation', yaxis_title='Nombre',
                  legend_title='Pluie demain')

# Affichage du graphique
fig11.show()

    
print("-"*100)
# Création du nuage de points avec Plotly Express
fig12 = px.scatter(
    weather_data.sample(2000),  # Utilisation d'un échantillon de 2000 lignes pour des raisons de visualisation
    x='Sunshine',
    title='fif12 - Ensoleillement avec RainTomorrow',
    color='RainTomorrow'
)
# Affichage du nuage de points
fig12.show()

print("-"*100)

# Remplacer les valeurs 'No' par 0 et 'Yes' par 1 dans la colonne 'RainTomorrow'
weather_data['RainTomorrow'] = weather_data['RainTomorrow'].replace(['No', 'Yes'], [0, 1])
# Remplacer les valeurs 'No' par 0 et 'Yes' par 1 dans la colonne 'RainToday'
weather_data['RainToday'] = weather_data['RainToday'].replace(['No', 'Yes'], [0, 1])
# Afficher les cinq premières lignes du DataFrame après les modifications
weather_data.head()

print("-"*100)

# Création de la fonction preprocess_evaporation
def preprocess_evaporation(value):
    if isinstance(value, str) and '-' in value:
        range_values = value.split('-')
        return (float(range_values[0]) + float(range_values[1])) / 2
    elif isinstance(value, str):
        return float(value)
    else:
        return value


weather_data['Evaporation'] = weather_data['Evaporation'].apply(preprocess_evaporation)
weather_data['Evaporation'] = weather_data['Evaporation'].astype(float)
mean = weather_data['Evaporation'].mean()
weather_data['Evaporation'].fillna(mean, inplace=True)
weather_data.info()

weather_data.isnull().sum()

# Calcul de la matrice de corrélation
corr_matrix = weather_data[numeric_col].corr()

# Sélectionner uniquement les colonnes numériques pour le calcul de la corrélation
numeric_cols = weather_data.select_dtypes(include=['float64', 'int64']).columns

# Calculer la matrice de corrélation
corr_matrix = weather_data[numeric_col].corr()

# Créer la heatmap avec la matrice de corrélation
fig13 = plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2g', linewidth=0.5, linecolor='black')
plt.title('fig13 - Heatmap of Correlation Matrix')
# Affichage du nuage de points
fig13.show()

'''
# Les variables 'MinTemp' et 'Temp9am' ont une forte corrélation positive de 0,898945, ce qui suggère que les températures du matin ont tendance 
# à être plus élevées lorsque les températures minimales sont plus élevées.

# Les variables 'Temp9am' et 'Temp3pm' ont également une forte corrélation positive de 0,848015, ce qui suggère que les températures ont tendance 
# à augmenter tout au long de la journée.

# Les variables 'MaxTemp' et 'Temp3pm' ont une forte corrélation positive de 0,970337, ce qui suggère que des températures maximales plus élevées 
# sont associées à des températures d'après-midi plus élevées.

# La variable 'Rainfall' a une corrélation positive modérée avec 'RainToday' (0,500997), ce qui indique que des précipitations plus élevées ont 
# tendance à être associées aux jours où il a plu.
'''

print("-"*100)

# Trouver les paires de caractéristiques avec une corrélation positive supérieure à 0,30
high_corr_features = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        correlation = corr_matrix.iloc[i, j]
        if correlation > 0.30:
            high_corr_features.append((corr_matrix.columns[i], corr_matrix.columns[j], correlation))

print("Paires de caractéristiques avec une corrélation positive > 0,30 :")
for feature1, feature2, correlation in high_corr_features:
    print(f"{feature1} - {feature2}: {correlation}")

print("-"*100)

# Visualiser la relation entre les températures à 9 heures du matin (Temp9am) et à 15 heures de l'après-midi (Temp3pm).
fig14 = scatter_plot = px.scatter(weather_data.sample(2000), x='Temp3pm', y='Temp9am', 
                          color='RainToday',  
                          title='fig14 - Scatter Plot of Temperature')

fig14.show()

print("-"*100)
print('Training Validation and Test Sets*3')
print("-"*100)

# Création d'un nuage de points pour visualiser la relation entre Rainfall et RainToday
fig15 = scatter_plot = px.scatter(weather_data.sample(2000), x='Rainfall', y='RainToday', 
                          color='RainTomorrow',  
                          title='fig15 - Scatter Plot of Rainfall and RainToday vs. RainTomorrow')

# Afficher le nuage de points
fig15.show()

print("-"*100)
print("-"*100)


# Sélection des colonnes catégorielles
category_col = weather_data.select_dtypes(include=['object']).columns.tolist()

# Initialisation de l'encodeur One-Hot
encoder = OneHotEncoder(handle_unknown='ignore')

# Application de l'encodage One-Hot
encoded_data = encoder.fit_transform(weather_data[category_col])

# Obtenir les noms des colonnes encodées
encoded_cols = encoder.get_feature_names_out(category_col)

# Assurez-vous que la forme de encoded_data correspond au nombre de colonnes encodées
print(encoded_data.shape)  # Doit être (nombre_de_lignes, nombre_de_colonnes_encodées)

# Création du DataFrame à partir des données encodées
weather_data_encoded = pd.DataFrame(encoded_data.toarray(), columns=encoded_cols)

# Affichez la forme et les premières lignes pour vérifier
print('nombre de colonnes et lignes de l\'encodage : ', weather_data_encoded.shape)
print('Premières lignes de l\'encodage : ', weather_data_encoded.head())

# Réinitialisation de l'index pour les deux DataFrames si nécessaire
weather_data.reset_index(drop=True, inplace=True)
weather_data_encoded.reset_index(drop=True, inplace=True)

# Suppression des colonnes catégorielles originales du DataFrame d'origine
weather_data = weather_data.drop(columns=category_col)

# Concaténation des données encodées avec le DataFrame d'origine
weather_data_final = pd.concat([weather_data, weather_data_encoded], axis=1)

# Affichage de la taille et des premières lignes du DataFrame final après transformation
print("Dimensions du DataFrame après encodage One-Hot:", weather_data_final.shape)
print("Ligne du dat final : \n", weather_data_final.head())

print("-"*100)
print("Backup du dataset dans 'WeatherUpdated.csv' -" * 5)
# Sauvegarde du nouveau DataFrame pour un usage ultérieur
weather_data_final.to_csv('WeatherUpdated.csv', index=False)
print("Le DataFrame mis à jour a été sauvegardé sous 'WeatherUpdated.csv'.")
weather_data.to_csv('WeatherUpdated.csv')
backup=pd.DataFrame(weather_data)
backup.sample()
print("-"*100)
print()
print("-"*100)
print("modèle1 Régression Logistique - " * 10)

# Suppression de la colonne Date
if 'Date' in weather_data.columns:
    weather_data = weather_data.drop(columns=['Date'])

# Suppression des colonnes catégorielles originales et concaténation des colonnes encodées
columns_to_drop = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
for column in columns_to_drop:
    if column in weather_data.columns:
        weather_data = weather_data.drop(columns=[column])
# weather_data = weather_data.drop(columns=category_col)
weather_data = pd.concat([weather_data, weather_data_encoded], axis=1)

# Séparation des données en caractéristiques et cible
X = weather_data.drop('RainTomorrow', axis=1)  # 'RainTomorrow' est la colonne cible
y = weather_data['RainTomorrow'].astype(float)  # Convertir en float 

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('-'*100)
# Vérification des types de données avant l'entraînement
print("Les types de données sont : \n" , weather_data.dtypes)
print('-'*100)


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
print(f'Accuracy du modèle 1: {accuracy:.4f}')
print(classification_report(y_test, y_pred))


print("- modèle 2 régression logistique -" *3)

# Division des données en caractéristiques (X) et cible (y)
X = weather_data.drop('RainTomorrow', axis=1)
y = weather_data['RainTomorrow'].astype(int)  # Assurez-vous que 'RainTomorrow' est correctement formaté

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle
# model = LogisticRegression(solver='liblinear')
# model.fit(X_train, y_train)

# Obtenir les noms des colonnes encodées
encoded_cols = list(encoder.get_feature_names_out(category_col))

# Création du DataFrame à partir des données encodées
weather_data_encoded = pd.DataFrame(encoded_data.toarray(), columns=encoded_cols)

# Réinitialisation de l'index si nécessaire
weather_data.reset_index(drop=True, inplace=True)
weather_data_encoded.reset_index(drop=True, inplace=True)

# Concaténation des données encodées avec le DataFrame d'origine
# weather_data_final = pd.concat([weather_data.drop(columns=category_col), weather_data_encoded], axis=1)

# Vérification des types de données après concaténation
# print("Les types des données concaténées : \n" , weather_data_final.dtypes)

# Supprimer les colonnes catégorielles d'origine du DataFrame
# weather_data.drop(columns=category_col, inplace=True)

# Afficher le nombre de lignes et de colonnes du DataFrame après les transformations
print(weather_data.shape)

# Afficher les premières lignes du DataFrame après les transformations
print(weather_data.head())

# Division des données en caractéristiques et cible, suivi par l'entraînement
X = weather_data_final.drop('RainTomorrow', axis=1)
y = weather_data_final['RainTomorrow'].astype(float)

# Séparation des données en caractéristiques (X) et cible (y)
X = weather_data_final.drop('RainTomorrow', axis=1)  # Supposons que 'RainTomorrow' est votre variable cible
y = weather_data_final['RainTomorrow'].astype(float)  # Assurez-vous que la variable cible est de type float pour la compatibilité

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle de régression logistique
model = LogisticRegression(solver='liblinear', random_state=42)

# Entraînement du modèle avec l'ensemble d'entraînement
# model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
# y_pred = model.predict(X_test)

# Calcul de l'exactitude du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy du modèle 2 : {accuracy:.4f}')

# Affichage du rapport de classification
print(classification_report(y_test, y_pred))

print("-" * 100)
print("modèle 3 Régression Logistique -"*10)


# Suppression des lignes où 'RainToday' ou 'RainTomorrow' sont manquants
weather_data.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# Fonction pour convertir les plages de valeurs en moyennes
def convert_range_to_mean(value):
    if '-' in str(value):
        low, high = map(float, value.split('-'))
        return (low + high) / 2
    try:
        return float(value)
    except ValueError:
        return np.nan

# Appliquer la fonction de conversion aux colonnes concernées
weather_data['Evaporation'] = weather_data['Evaporation'].apply(convert_range_to_mean)

# Remplacement des valeurs manquantes dans 'Evaporation' par la moyenne
weather_data['Evaporation'] = weather_data['Evaporation'].fillna(weather_data['Evaporation'].mean())

# Séparation des caractéristiques et des cibles
X = weather_data.drop(['RainTomorrow'], axis=1)
y = weather_data['RainTomorrow'].replace(['No', 'Yes'], [0, 1]).astype(int)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Sélection des colonnes numériques et catégorielles
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns

# Pipelines pour le prétraitement
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Création du pipeline de modélisation
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

# Entraînement du modèle
model.fit(X_train, y_train)

# Évaluation du modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy modèle 3 : {accuracy:.4f}')

# Affichage du rapport de classification
print(classification_report(y_test, y_pred))

'''

print("-" * 100)
print("modèle 4 Random Forest -" * 10)
# Cette version du code intègre le prétraitement et la modélisation dans un pipeline

# Définir le chemin absolu du fichier weatherAUS.csv qui fournit des données météorologiques spécifiques pour certaines de villes d'Australie
chemin_weatherAUS = "C:\\Users\\Utilisateur\\Mon Drive\\PROJET_METEO_DEC23_DS\\PRODUCTION\\Patrick\\data_meteo_patrick\\weatherAUS.csv"

# Charger le fichier weatherAUS.csv qui contient les données des villes d'Australie sous un angle géographique et démographique
weather_data = pd.read_csv(chemin_weatherAUS)

# Conversion des plages de valeurs en moyennes pour 'Evaporation' et autres colonnes similaires
def convert_range_to_mean(value):
    if '-' in str(value):
        low, high = map(float, value.split('-'))
        return (low + high) / 2
    try:
        return float(value)
    except ValueError:
        return np.nan

# Appliquer la fonction de conversion à la colonne 'Evaporation'
weather_data['Evaporation'] = weather_data['Evaporation'].apply(convert_range_to_mean)

# Suppression des lignes où 'RainToday' ou 'RainTomorrow' sont manquants
weather_data.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# Remplacer 'No' par 0 et 'Yes' par 1 pour 'RainToday' et 'RainTomorrow'
weather_data['RainToday'] = weather_data['RainToday'].replace({'No': 0, 'Yes': 1})
weather_data['RainTomorrow'] = weather_data['RainTomorrow'].replace({'No': 0, 'Yes': 1})

# Convertir explicitement les colonnes en type entier après le remplacement
weather_data['RainToday'] = weather_data['RainToday'].astype(int)
weather_data['RainTomorrow'] = weather_data['RainTomorrow'].astype(int)

# Fonction pour convertir les plages de valeurs en moyennes
def convert_range_to_mean(value):
    if '-' in str(value):
        low, high = map(float, value.split('-'))
        return (low + high) / 2
    try:
        return float(value)
    except ValueError:
        return np.nan

# Appliquer la fonction de conversion aux colonnes concernées
weather_data['Evaporation'] = weather_data['Evaporation'].apply(convert_range_to_mean).fillna(weather_data['Evaporation'].mean())

# Préparation des données pour le modèle
X = weather_data.drop(['RainTomorrow'], axis=1)
y = weather_data['RainTomorrow']

# Encodage des variables catégorielles et imputation pour les valeurs manquantes
numeric_features = weather_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Division des données
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construction du pipeline du modèle
from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Entraînement du modèle
rf_pipeline.fit(X_train, y_train)

# Évaluation du modèle
from sklearn.metrics import accuracy_score, classification_report

y_pred_rf = rf_pipeline.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Accuracy of Random Forest model: {accuracy_rf:.4f}')
print(classification_report(y_test, y_pred_rf))

print("-" * 100)
print("modèle 5 XGBoost -" * 10)


# Sélection des caractéristiques et de la cible
X = weather_data.drop(['RainTomorrow', 'Date'], axis=1)  # Exclure 'Date' et 'RainTomorrow'
y = weather_data['RainTomorrow'].astype(int)

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identification des colonnes catégorielles et numériques
categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Création des transformateurs pour les colonnes numériques et catégorielles
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Préprocesseur qui applique les transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

# Création du pipeline de modélisation avec XGBClassifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Entraînement du modèle
model_pipeline.fit(X_train, y_train)

# Prédiction sur l'ensemble de test avec le modèle XGBoost
y_pred = model_pipeline.predict(X_test)

# Calcul de l'exactitude pour le modèle XGBoost
accuracy_XGB = accuracy_score(y_test, y_pred)
print(f'Accuracy du modèle XGBoost : {accuracy_XGB:.4f}')

# Affichage du rapport de classification pour le modèle XGBoost
print(classification_report(y_test, y_pred))


print("-" * 100)
print("modèle 6 SVM -" * 10)


# Sélection des caractéristiques et de la cible
X = weather_data.drop('RainTomorrow', axis=1)
y = weather_data['RainTomorrow'].astype(int)  # Assurez-vous que 'RainTomorrow' est de type int

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prétraitement: Imputation des valeurs manquantes et normalisation pour les variables numériques, encodage OneHot pour les catégoriques
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Création du pipeline de modélisation avec SVM
model_pipeline = make_pipeline(preprocessor, SVC(kernel='linear', C=1, random_state=42))

# Entraînement du modèle SVM
model_pipeline.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model_pipeline.predict(X_test)

# Évaluation du modèle
accuracy_svm = accuracy_score(y_test, y_pred)
print(f'Accuracy of SVM model: {accuracy_svm:.4f}')
print(classification_report(y_test, y_pred))



print("-" * 100)
print("modèle 7 KNN -" * 10)


# Séparation des données en caractéristiques et étiquettes
X = weather_data.drop('RainTomorrow', axis=1)  # Supposons que 'RainTomorrow' soit la cible
y = weather_data['RainTomorrow'].astype(int)  # Assurez-vous que la cible est numérique

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prétraitement: Normalisation des caractéristiques numériques et encodage des caractéristiques catégorielles
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Création du modèle KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Vous pouvez ajuster le nombre de voisins

# Pipeline pour le prétraitement et le modèle KNN
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', knn)])

# Entraînement du modèle
pipeline.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = pipeline.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))



print("-" * 100)
print("modèle 8 K-MEANS -" * 10)

# Sélection des caractéristiques numériques uniquement 
X = weather_data.select_dtypes(include=['int64', 'float64'])

# 'Evaporation' est une colonne avec des plages de valeurs comme '4-6'
# Remplacement des plages par la moyenne des valeurs
# la fonction convert_range_to_mean est définie dans votre code
X['Evaporation'] = X['Evaporation'].apply(lambda x: convert_range_to_mean(x) if isinstance(x, str) else x)

# Conversion des colonnes à des types numériques
X = X.apply(pd.to_numeric, errors='coerce')

# Création et configuration du pipeline pour K-Means
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),  # Remplacement des NaN par la moyenne de chaque colonne
    StandardScaler(),  # Mise à l'échelle des caractéristiques
    KMeans(n_clusters=3, random_state=42)  # Application de K-Means avec 3 clusters
)

# Ajustement du pipeline sur les données
pipeline.fit(X)

# Récupération des étiquettes de cluster pour chaque point de données
labels = pipeline.predict(X)

# Ajout des étiquettes de cluster au DataFrame original pour l'analyse
weather_data['Cluster'] = labels

# Sélectionnez uniquement les colonnes numériques pour calculer la moyenne
numeric_cols = weather_data.select_dtypes(include=[np.number]).columns
grouped_data = weather_data[numeric_cols].groupby('Cluster').mean()

# Affichez la moyenne des caractéristiques numériques pour chaque cluster
print(grouped_data)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy modèle 8 : {accuracy:.4f}')
print(classification_report(y_test, y_pred))



print("-" * 100)
print("modèle 9 régression linéaire -" * 10)

# Charger les données
chemin_weatherAUS = "C:\\Users\\Utilisateur\\Mon Drive\\PROJET_METEO_DEC23_DS\\PRODUCTION\\Patrick\\data_meteo_patrick\\weatherAUS.csv"
weather_data = pd.read_csv(chemin_weatherAUS)

# Prétraitement
# Supprimer les lignes où la cible est NaN
weather_data.dropna(subset=['MaxTemp'], inplace=True)

# Sélection des caractéristiques et de la cible
X = weather_data.drop('MaxTemp', axis=1)
y = weather_data['MaxTemp']

# Séparation des colonnes numériques et catégorielles
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Pipeline pour les transformations numériques et catégorielles
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputation par la moyenne pour les NaN
    ('scaler', StandardScaler())  # Mise à l'échelle des caractéristiques
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Gestion des NaN pour catégorielles
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encodage One-Hot
])

# Assemblage dans un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Création du pipeline final incluant le préprocesseur et le modèle
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model_pipeline.fit(X_train, y_train)

# Prédiction et évaluation du modèle
y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'R^2: {r2:.2f}')

# Accéder aux coefficients du modèle de régression linéaire
coefficients = model_pipeline.named_steps['regressor'].coef_
print("Coefficients du modèle 9 de régression linéaire :")
print(coefficients)
'''




print("-" * 100)
print("modèle 10 Réseaux de Neurones (Deep Learning) -" * 3)

'''
# Exemple de transformer pour convertir les plages en moyennes
class RangeToMeanTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for column in X.columns:
            X[column] = X[column].apply(lambda x: self.convert_range_to_mean(x))
        return X

    def convert_range_to_mean(self, value):
        if isinstance(value, str) and '-' in value:
            low, high = map(float, value.split('-'))
            return (low + high) / 2
        return value
'''

# Prétraitement pour les données numériques et catégorielles
numeric_features = ['list', 'of', 'numeric', 'columns']  # Mettez ici vos colonnes numériques
categorical_features = ['list', 'of', 'categorical', 'columns']  # Mettez ici vos colonnes catégorielles

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combinez les transformateurs
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Définir le modèle de réseaux de neurones
def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = XGBClassifier(build_fn=create_model, input_shape=[X_train.shape[1]], epochs=10, batch_size=10, verbose=1)

# Créer le pipeline complet
nn_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Entraînement du modèle
nn_pipeline.fit(X_train, y_train)

# Évaluation du modèle
nn_accuracy = nn_pipeline.score(X_test, y_test)
print(f'Accuracy of Neural Network: {nn_accuracy:.4f}')


'''
'''
print("-" * 100)
print("modèle 11 Gradient Boosting Machines (GBM) -" * 3)

# Pipeline de prétraitement et GBM
gbm_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
)

# Entraînement du modèle
gbm_pipeline.fit(X_train, y_train)

# Évaluation du modèle
gbm_accuracy = gbm_pipeline.score(X_test, y_test)
print(f'Accuracy of Gradient Boosting Machine: {gbm_accuracy:.4f}')



print("-" * 100)
print("modèle 12 Régression Linéaire Régularisée - Ridge Regression -" * 3)

# Pipeline de prétraitement et Ridge Regression
ridge_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    Ridge(alpha=1.0)
)

# Entraînement du modèle
ridge_pipeline.fit(X_train, y_train)

# Évaluation du modèle
ridge_score = ridge_pipeline.score(X_test, y_test)
print(f'R^2 score of Ridge Regression: {ridge_score:.4f}')




print("-" * 100)
print("modèle 13 Régression Linéaire Régularisée - Lasso Regression -" * 3)

# Pipeline de prétraitement et Lasso Regression
lasso_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    Lasso(alpha=0.1)
)

# Entraînement du modèle
lasso_pipeline.fit(X_train, y_train)

# Évaluation du modèle
lasso_score = lasso_pipeline.score(X_test, y_test)
print(f'R^2 score of Lasso Regression: {lasso_score:.4f}')




print("-" * 100)
print("modèle 14 Réseaux de Neurones Récurrents (RNN) -" * 3)


# Supposons que X_train et y_train soient déjà formatés correctement pour un RNN,
# c'est-à-dire avec la forme [échantillons, pas de temps, caractéristiques].

model = Sequential()
model.add(SimpleRNN(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(SimpleRNN(50, return_sequences=False))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

# Entraînement du modèle
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Évaluation du modèle
loss = model.evaluate(X_test, y_test)
print(f'Test Loss of RNN: {loss:.4f}')

'''
print("graphique global -" * 10)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Supposons que vous ayez des prédictions de plusieurs modèles
predictions = {
    'Model 1': model1_predictions,
    'Model 2': model2_predictions,
    'Model 3': model3_predictions,
    # Ajoutez autant de modèles que nécessaire
}

# La variable 'targets' contient les vraies valeurs cibles
targets = y_test

# Création d'une figure pour les sous-graphiques
fig30, axes = plt.subplots(nrows=1, ncols=len(predictions), figsize=(15, 5))

for ax, (title, preds) in zip(axes, predictions.items()):
    # Calcul de la matrice de confusion pour chaque modèle
    conf_mat = confusion_matrix(targets, preds)
    
    # Affichage de la matrice de confusion avec Seaborn
    sns.heatmap(conf_mat, annot=True, linewidth=0.5, linecolor='Blue', cmap='Oranges', ax=ax)
    ax.set_title("fig30", title)
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Targets")

plt.tight_layout()
plt.show()
'''

'''
# Liste des colonnes catégorielles
category_col = weather_data.select_dtypes(include=['object']).columns.tolist()

# Initialisation de l'encodeur One-Hot
encoder = OneHotEncoder(handle_unknown='ignore')

# Application de l'encodage One-Hot
encoded_data = encoder.fit_transform(weather_data[category_col])

# Obtenir les noms des nouvelles colonnes après encodage
encoded_cols = list(encoder.get_feature_names_out(category_col))

# Création d'un DataFrame à partir des données encodées
# weather_data_encoded = pd.DataFrame(encoded_data, columns=encoded_cols)

# Réinitialisation de l'index pour la concaténation
weather_data.reset_index(drop=True, inplace=True)
weather_data_encoded.reset_index(drop=True, inplace=True)

# Concaténation des données encodées avec le DataFrame d'origine (moins les colonnes catégorielles)
weather_data = pd.concat([weather_data.drop(columns=category_col), weather_data_encoded], axis=1)

# Affichage de la taille du DataFrame après transformation
print("Dimensions du DataFrame après encodage One-Hot:", weather_data.shape)

# Affichage des premières lignes pour vérifier
print(weather_data.head())

print("-" * 100)

# Sauvegarde du nouveau DataFrame pour un usage ultérieur
weather_data.to_csv('WeatherUpdated.csv', index=False)
print("Le DataFrame mis à jour a été sauvegardé sous 'WeatherUpdated.csv'.")

# Affichage d'un échantillon de données pour vérification
print("Aperçu des données sauvegardées :")
print(pd.read_csv('WeatherUpdated.csv').sample(5))




# Réinitialiser les index du DataFrame
weather_data.reset_index(drop=True, inplace=True)

# Vérification de la présence de la colonne 'RainTomorrow'
if 'RainTomorrow' in weather_data.columns:
    print("'RainTomorrow' est présent dans weather_data.")
else:
    print("'RainTomorrow' n'est pas présent dans weather_data.")
    
# Diviser les données en ensembles d'entraînement, de validation, et de test
train_val_df, test_df = train_test_split(weather_data, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

print("-" * 100)
# Vérification après la division
if 'RainTomorrow' in train_df.columns:
    print("'RainTomorrow' est présent dans train_df.")
else:
    print("'RainTomorrow' n'est pas présent dans train_df.")
print("-" * 100)

# Diviser les données en 80% pour l'ensemble d'entraînement et 20% pour l'ensemble de test
train_val_df, test_df = train_test_split(weather_data, test_size=0.2, random_state=42)

# Diviser l'ensemble d'entraînement restant en 75% pour l'ensemble d'entraînement et 25% pour l'ensemble de validation
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

# Affichage des dimensions des ensembles de données
print("test_df", test_df.shape)
print("train_df", train_df.shape)
print("val_df", val_df.shape)

print("-" * 100)
# Afficher les noms de colonnes du DataFrame train_df
print(train_df.columns)
print("-" * 100)
# Vérifier si la colonne 'RainTomorrow' est présente dans train_df
if 'RainTomorrow' in train_df.columns:
    # Effectuer les opérations nécessaires sur train_df
    train_inputs, train_targets = train_df[category_col].copy(), train_df['RainTomorrow'].copy()
else:
    print("La colonne 'RainTomorrow' n'est pas présente dans le DataFrame train_df.")
print("-" * 100)

# Séparation des colonnes d'entrée et de sortie
input_col = weather_data.columns[~weather_data.columns.isin(['Date', 'RainTomorrow'])]
target_col = 'RainTomorrow'

print("-" * 100)
# Afficher les noms de colonnes du DataFrame train_df
print(train_df.columns)
print("-" * 100)

print("-" * 100)
# Vérification de la Présence de la Colonne 'RainTomorrow'
print('la colonne "RainTomorrow" est présente dans le dataset : \n' , weather_data.columns)  # Vérifiez que 'RainTomorrow' est listé
print("-" * 100)

# Division des données en ensembles d'entraînement, de validation et de test
train_inputs, train_targets = train_df[input_col].copy(), train_df[target_col].copy()
val_inputs, val_targets = val_df[input_col].copy(), val_df[target_col].copy()
test_inputs, test_targets = test_df[input_col].copy(), test_df[target_col].copy()

# Entraînement du modèle de régression logistique
model = LogisticRegression(solver='liblinear')
# model.fit(train_inputs, train_targets)

print("-" * 100)

# Affichage des coefficients du modèle
model_coef = model.coef_
train_input_col_coef = train_inputs.columns
coef_df = pd.DataFrame(train_input_col_coef, columns=['feature'])
coef_df['model_coef'] = model_coef.reshape(-1)
print("-" * 100)


print("-" * 100)
# Affichage des coefficients sous forme de graphiques
fig15 = plt.figure(figsize=(10, 25))
sns.barplot(data=coef_df, x='model_coef', y='feature')
# Affichage du nuage de points
fig15.show()
print("-" * 100)

print("-" * 100)
# Analyse des coefficients importants
top_20_model_coef = coef_df.sort_values('model_coef', ascending=False).head(20)
px.bar(top_20_model_coef, x='model_coef', y='feature', title='TOP 20 Model Coef with features')
print("-" * 100)


print("-" * 100)
# Prédiction et évaluation sur les données d'entraînement
train_predict = model.predict(train_inputs)
train_targets # Les vraies valeurs
print("-" * 100)

# Utilisation du modèle entraîné pour faire des prédictions sur de nouvelles données et évaluer ses performances.
# Prédiction des probabilités sur l'ensemble d'entraînement
train_probs = model.predict_proba(train_inputs)

# Construction d'un DataFrame pour les probabilités
No_probs = train_probs[:, 0]
Yes_probs = train_probs[:, 1]
model_probs = pd.DataFrame({'No_probs': No_probs, 'Yes_probs': Yes_probs})

print("-" * 100)
# Visualisation des probabilités prédites
px.histogram(model_probs.head(100), x='No_probs', title='Min temp with RainTomorrow', y='Yes_probs')
print("-" * 100)

# Conversion des probabilités en chaînes de caractères et création d'un DataFrame
train_probs = train_probs.astype(str)
train_probs = pd.DataFrame(train_probs)

# Matrice de confusion pour l'ensemble d'entraînement
confusion_mat_values = confusion_matrix(train_targets, train_predict, normalize='true')
fig16 = sns.heatmap(confusion_mat_values, annot=True, linewidth=0.5, linecolor='Blue', cmap='Blues')
plt.xlabel("Prédictions")
plt.ylabel("Cibles")
# Affichage du nuage de points
fig16.show()

# Prédiction et évaluation sur l'ensemble de validation
val_predict = model.predict(val_inputs)
accuracy_scr_val_set = accuracy_score(val_targets, val_predict)
print("Score d'exactitude sur l'ensemble de validation : {:.2f} %".format((accuracy_scr_val_set * 100)))
confusion_mat_val_set = confusion_matrix(val_targets, val_predict, normalize='true')
fig17 = sns.heatmap(confusion_mat_val_set, annot=True, linewidth=0.5, linecolor='Blue', cmap='Oranges')
plt.xlabel("Prédictions_val_set")
plt.ylabel("Cibles_val_set")
# Affichage du nuage de points
fig17.show()

# Prédiction et évaluation sur l'ensemble de test
test_predict = model.predict(test_inputs)
accuracy_scr_test_set = accuracy_score(test_targets, test_predict)
print("Score d'exactitude sur l'ensemble de test : {:.2f} %".format((accuracy_scr_test_set * 100)))
confusion_mat_test_set = confusion_matrix(test_targets, test_predict, normalize='true')
fig18 = sns.heatmap(confusion_mat_test_set, annot=True, linewidth=0.5, linecolor='Blue', cmap='Purples')
plt.xlabel("Prédictions_test_set")
plt.ylabel("Cibles_test_set")
# Affichage du nuage de points
fig18.show()

print("-" * 100)
# Calcul de l'exactitude du modèle de regression logistique
accuracy_scr = accuracy_score(train_targets, train_predict)
print("Score d'exactitude : {:.2f} %".format((accuracy_scr * 100)))
print('score = ' , accuracy_scr)

# Score d'exactitude du modèle de regression logistique
accuracy_scr = accuracy_score(train_targets, train_predict)

# Création du graphique 
fig19, ax = plt.subplots(figsize=(6, 4))
ax.bar(['Score d\'exactitude'], [accuracy_scr * 100], color='skyblue')
ax.set_title('Score d\'exactitude du modèle de régression logistique')
ax.set_ylabel('Exactitude (%)')
ax.set_ylim(0, 100)
ax.text(0, accuracy_scr * 100 + 1, f'{accuracy_scr * 100:.2f}%', ha='center')

# Sauvegarde de fig19 dans un fichier temporaire
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
fig19.savefig(temp_file.name)
plt.close(fig19)
# Ouverture du fichier temporaire dans le navigateur web par défaut
webbrowser.open('file://' + os.path.realpath(temp_file.name))
# Décommenter la ligne de code suivante si on ne souhaite pas conserver le fichier temporaire
# os.unlink(temp_file.name)

print("-" * 100)
print("Prédictions -" * 20)
print("-" * 100)

# Utilisation de la méthode predict_proba
train_probs = model.predict_proba(train_inputs)
print(model.classes_)
print(train_probs)

# Création des colonnes pour les probabilités de chaque classe
No_probs = train_probs[:, 0]
Yes_probs = train_probs[:, 1]
model_probs = pd.DataFrame({'No_probs': No_probs, 'Yes_probs': Yes_probs})
print(model_probs)

# Affichage d'un histogramme des probabilités
# Création de l'histogramme avec Plotly
fig20 = px.histogram(model_probs.head(100), x='No_probs', y='Yes_probs',
                   title='Probabilités de Pluie Oui ou Non Demain')

# Affichage du graphique dans le navigateur web
fig20.show()


# Conversion des probabilités en chaînes de caractères et création d'un DataFrame
train_probs = train_probs.astype(str)
train_probs = pd.DataFrame(train_probs)
print(train_probs)

# Calcul de l'accuracy
accuracy_score_train = accuracy_score(train_targets, train_predict)
print("Accurecy Score : {:.2f} %".format((accuracy_score_train * 100)))

# Matrice de confusion
confusion_mat_values = confusion_matrix(train_targets, train_predict, normalize='true')
print(confusion_mat_values)

# Visualisation de la matrice de confusion
fig21 = sns.heatmap(confusion_mat_values, annot=True, linewidth=0.5, linecolor='Blue', cmap='Blues')
plt.xlabel("Predictions")
plt.ylabel("Targets")
# Affichage du nuage de points
fig21.show()

# Interprétation de la matrice de confusion
print("-" * 100)
print("description textuelle de la matrice de confusion à faire ici - " * 3)
print("-" * 100)


# Prédiction sur l'ensemble de validation
val_predict = model.predict(val_inputs)
accuracy_score_val_set = accuracy_score(val_targets, val_predict)
print("Accurecy Score : {:.2f} %".format((accuracy_score_val_set * 100)))
confusion_mat_val_set = confusion_matrix(val_targets, val_predict, normalize='true')
print(confusion_mat_val_set)
fig15 = sns.heatmap(confusion_mat_val_set, annot=True, linewidth=0.5, linecolor='Blue', cmap='Oranges')
plt.xlabel("Predictions_val_set")
plt.ylabel("Targets_val_set")
# Affichage du nuage de points
fig15.show()

# Prédiction sur l'ensemble de test
test_predict = model.predict(test_inputs)
accuracy_score_test_set = accuracy_score(test_targets, test_predict)
print("Accurecy Score : {:.2f} %".format((accuracy_score_test_set * 100)))
confusion_mat_test_set = confusion_matrix(test_targets, test_predict, normalize='true')
print(confusion_mat_test_set)
fig16 = sns.heatmap(confusion_mat_test_set, annot=True, linewidth=0.5, linecolor='Blue', cmap='Purples')
plt.xlabel("Predictions_test_set")
plt.ylabel("Targets_test_set")
# Affichage du nuage de points
fig16.show()
'''