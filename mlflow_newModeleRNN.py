

# Étape 1: Chargement des Données et dépendances
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

# Chargement des données
chemin_weatherAUS = 'weatherAUS.csv'
weather_data = pd.read_csv(chemin_weatherAUS)

# Afficher la colonne 'RainTomorrow' et vérifier sa présence
print(weather_data.columns)  # Pour déboguer

# Étape 2: Nettoyage des Données

# Analyse de la Significativité Statistique des Variables Climatiques sur la Prédiction des Précipitations : Résultats du Test du Chi Carré
# Sélection des colonnes catégorielles pour le test du Chi Carré
columns_to_test = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

# Encodage des variables catégorielles
label_encoders = {}
for column in columns_to_test:
    if weather_data[column].dtype == 'object':
        le = LabelEncoder()
        weather_data[column] = le.fit_transform(weather_data[column].astype(str))
        label_encoders[column] = le

# La variable cible
y = LabelEncoder().fit_transform(weather_data['RainTomorrow'].astype(str))

# Application du test du Chi Carré
chi_scores = chi2(weather_data[columns_to_test], y)

# Affichage des résultats du test chi carré
for col, p_value in zip(columns_to_test, chi_scores[1]):
    print(f"Chi-square test p-value for {col}: {p_value}")

# Suppression des colonnes en fonction des résultats du test du Chi Carré
# Exemple : si la p-value pour 'WindGustDir' est supérieure à 0.05, supprimez-la
# Ce seuil de 0.05 est arbitraire et peut être ajusté selon votre critère de significativité
columns_to_drop = [col for col, p_value in zip(columns_to_test, chi_scores[1]) if p_value > 0.05]
weather_data.drop(columns=columns_to_drop, inplace=True)

# Les valeurs de p du test du chi carré pour différentes variables.
# les valeurs de p sont toutes très proches de zéro, ce qui indique une forte significativité statistique.
# Ces résultats suggèrent fortement qu'il existe des associations significatives entre la localisation, la direction du vent 
# à différents moments de la journée, la pluie aujourd'hui, la pluie demain et la variable cible RainTomorrow.
# Chi-square test p-value for Location: 0.0
# Chi-square test p-value for WindGustDir: 0.0
# Chi-square test p-value for WindDir9am: 0.0
# Chi-square test p-value for WindDir3pm: 7.573688973424688e-275
# Chi-square test p-value for RainToday: 0.0
# Chi-square test p-value for RainTomorrow: 0.0
# On va faire le choix de supprimmer les colonnes des variables qui n'ont pas de corrélation forte d'après le test Chi-carré
    
weather_data.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am'], inplace=True)
weather_data.fillna(method='ffill', inplace=True)

# Étape 3: Remplacement des valeurs manquantes
num_vars = weather_data.select_dtypes(include=['float64', 'int64']).columns
cat_vars = weather_data.select_dtypes(include=['object']).columns
weather_data[num_vars] = weather_data[num_vars].fillna(weather_data[num_vars].median())
weather_data[cat_vars] = weather_data[cat_vars].fillna(weather_data[cat_vars].mode().iloc[0])

# Encodage de 'RainToday' et 'RainTomorrow'
weather_data['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
weather_data['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

# Étape 4: Séparation des Données
X = weather_data.drop('RainTomorrow', axis=1)
y = weather_data['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mise à jour des listes des colonnes numériques et catégorielles après le nettoyage
num_vars = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
cat_vars = [col for col in X_train.columns if X_train[col].dtype == 'object']

# Étape 5: Construction et Entraînement du Modèle
# Prétraitement et modèle
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())]), num_vars),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_vars)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(solver='liblinear', max_iter=1000))])

# Entraînement du modèle
model.fit(X_train, y_train)

# Étape 6: Évaluation du Modèle
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("NNR 3 - "*20)
# Points clés de cette approche :

# Augmentation des unités et couches : Architecture plus profonde avec plus d'unités par couche pour capturer des relations plus complexes.
# Batch Normalization : Utilisée pour normaliser les activations de chaque couche, permettant une convergence plus rapide et une performance accrue.
# Dropout : Ajouté pour prévenir le surajustement en "éteignant" aléatoirement des neurones pendant l'entraînement.
# Early Stopping : Pour arrêter l'entraînement lorsque la performance sur l'ensemble de validation cesse de s'améliorer.
# MLflow : Pour le suivi des expériences et la comparaison des performances entre les différentes itérations.    

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import mlflow
from mlflow import MlflowClient
import mlflow.tensorflow

# Prétraitement des données
df = pd.read_csv('df_full_cosin.csv', sep=',', header=0)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by='Date')
df = df.drop(["Date", "Location"], axis=1)
df["RainTomorrow"] = df["RainTomorrow"].astype(np.int8)

target = df['RainTomorrow']
data = df.drop('RainTomorrow', axis=1)
encoder = LabelEncoder()
target = encoder.fit_transform(target)

# Rééquilibrage des données avec SMOTE
smote = SMOTE(random_state=42)
data_resampled, target_resampled = smote.fit_resample(data, target)

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(data_resampled, target_resampled, test_size=0.25, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Architecture du modèle
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_model((X_train_scaled.shape[1],))

# Compilation
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraînement
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=64,
                    validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Évaluation
predictions = model.predict(X_test_scaled)
predictions = (predictions > 0.5).astype(int)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Logging avec MLflow
mlflow.tensorflow.autolog()
with mlflow.start_run(run_name="rnn_xgb2_ameliore"):
    mlflow.log_params({"epochs": 100, "batch_size": 64, "layers": "2+BN+Dropout", "units": "128-64", "activation": "relu"})
    mlflow.log_metrics({"accuracy": max(history.history['val_accuracy'])})





print("NNR 4 - "*20)
# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint
import tensorflow as tf
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


# Vérifier les GPU disponibles
print("GPUs disponibles: ", tf.config.experimental.list_physical_devices('GPU'))

from sklearn.metrics import classification_report

# Test pour s'assurer que TensorFlow peut accéder au GPU
with tf.device('/gpu:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
print(f"Résultat de la multiplication matricielle sur GPU:\n{c}")

# Chargement des données
chemin_weatherAUS = 'weatherAUS.csv'
weather_data = pd.read_csv(chemin_weatherAUS)
weather_data.dropna(subset=['RainTomorrow'], inplace=True)
weather_data['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
weather_data['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)


# Étape 2: Nettoyage des Données

# Analyse de la Significativité Statistique des Variables Climatiques sur la Prédiction des Précipitations : Résultats du Test du Chi Carré
# Sélection des colonnes catégorielles pour le test du Chi Carré selon les commentaires et observations obtenus par l'équipe et moi-même

columns_to_test = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
label_encoder = LabelEncoder()
for column in columns_to_test + ['RainTomorrow']:
    weather_data[column] = label_encoder.fit_transform(weather_data[column].astype(str))

# Application du test du Chi Carré
X_chi2 = weather_data[columns_to_test]
y_chi2 = weather_data['RainTomorrow']
chi_scores = chi2(X_chi2, y_chi2)

# Affichage des résultats du test du Chi Carré
print("\nRésultats du test Chi-square :")
for col, p_value in zip(columns_to_test, chi_scores[1]):
    print(f"{col}: p-value = {p_value}")

# Suppression des colonnes en fonction des résultats du test du Chi Carré
#  p-value pour 'WindGustDir' avec un seuil de 0.05 
columns_to_drop = [col for col, p_value in zip(columns_to_test, chi_scores[1]) if p_value > 0.05]
weather_data.drop(columns=columns_to_drop, inplace=True)


# Sauvegarde du DataFrame traité
weather_data.to_csv('weather_data_processed.csv', index=False)

# Prétraitement et 
# Utiliser un échantillon plus petit pour réduire le temps de calcul de 10% des données
# Ce code réduit la taille du jeu de données, simplifie le modèle RandomForestClassifier en limitant 
# le nombre d'estimateurs et la profondeur maximale, et réduit l'espace des hyperparamètres et 
# le nombre d'itérations dans RandomizedSearchCV. Ces ajustements devraient accélérer le processus d'entraînement 
# et de validation croisée.

weather_data_sample = weather_data.sample(frac=0.1, random_state=42)
X = weather_data_sample.drop('RainTomorrow', axis=1)
y = weather_data_sample['RainTomorrow']

# Séparation en jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Prétraitement
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Modèle
model = ImbPipeline(steps=[('preprocessor', preprocessor),
                           ('sampling', SMOTE(random_state=42)),
                           ('classifier', RandomForestClassifier(random_state=42, n_estimators=1000, max_depth=None))])


# Réduction de l'espace des hyperparamètres
param_dist = {
    "classifier__n_estimators": randint(100, 1000),  # Nombre d'arbres
    "classifier__max_depth": [10, 20, 30, None],    # Profondeur maximale
    "classifier__min_samples_split": randint(2, 10), # Min échantillons pour diviser un nœud
    "classifier__min_samples_leaf": randint(1, 4),   # Min échantillons dans un nœud feuille
    "classifier__bootstrap": [True, False]           # Méthode de sélection des échantillons
}

# Randomized search sur les hyperparamètres
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1)



# Entraînement et évaluation
random_search.fit(X_train, y_train) 
y_pred = random_search.predict(X_test) 

# Meilleurs paramètres
print("Meilleurs paramètres: ", random_search.best_params_)

# Utiliser le meilleur modèle trouvé pour faire des prédictions sur le jeu de test
y_pred = random_search.predict(X_test)

# Affichage du rapport
print("Rapport de classification :\n", classification_report(y_test, y_pred)) 




# print("NNR 5 - "*20)