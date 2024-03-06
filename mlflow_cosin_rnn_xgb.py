
print("NNR 1 - "*20)
# Code de Mogens
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import mlflow
from mlflow import MlflowClient
import mlflow.keras

# Define tracking_uri and set experiment
client = MlflowClient(tracking_uri="http://127.0.0.1:8888")
mlflow.set_experiment("Meteo_cosin_rnn_xgb")

# Import database
df = pd.read_csv('df_full_cosin.csv', sep=',', header=0)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by='Date')
df = df.drop(["Date", "Location"], axis=1)
df["RainTomorrow"] = df["RainTomorrow"].astype(np.int8)

# Preprocessing
target = df['RainTomorrow']
data = df.drop('RainTomorrow', axis=1)
encoder = LabelEncoder()
target = encoder.fit_transform(target)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, shuffle=False)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Define the model architecture
inputs = Input(shape=(X_train_scaled.shape[1],), name="Input")
x = Dense(units=24, activation="tanh", name="Couche_1")(inputs)
x = Dense(units=16, activation="tanh", name="Couche_2")(x)
x = Dense(units=10, activation="tanh", name="Couche_3")(x)
x = Dense(units=6, activation="tanh", name="Couche_4")(x)
outputs = Dense(units=1, activation='sigmoid', name="Couche_5")(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.01,
                               patience=5,
                               verbose=1,
                               restore_best_weights=True,
                               mode='min')

# Train the model with class weights
history = model.fit(X_train_scaled, y_train,
                    epochs=500,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1,
                    class_weight=class_weights_dict)

# Evaluate the model
predictions = model.predict(X_test_scaled)
predictions = (predictions > 0.5).astype(int)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Log the model and results to MLflow
with mlflow.start_run(run_name="rnn_xgb1"):
    mlflow.log_params({
        "epochs": 500,
        "batch_size": 32,
        "layers": "4",
        "units": "24-16-10-6",
        "activation": "tanh"
    })
    mlflow.log_metrics({
        "val_accuracy": max(history.history['val_accuracy']),
        "val_loss": min(history.history['val_loss'])
    })
    # Log the Keras model
    mlflow.keras.log_model(model, "model")


print("NNR 2 - "*20)
# Ce code introduit les modifications suivantes par rapport à la version originale :
# Rééquilibrage des classes avec SMOTE pour traiter le déséquilibre des classes.
# Introduction de couches Dropout pour aider à réduire le surajustement en ajoutant de la régularisation.
# Augmentation du nombre d'unités dans les couches cachées pour permettre au modèle de capturer des relations plus complexes.
# Utilisation de EarlyStopping pour arrêter l'entraînement si le modèle ne s'améliore plus sur l'ensemble de validation.

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import mlflow
from mlflow import MlflowClient
import mlflow.keras

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

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data_resampled, target_resampled, test_size=0.25, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définition de l'architecture du modèle
inputs = Input(shape=(X_train_scaled.shape[1],))
x = Dense(64, activation='relu')(inputs)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle avec EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Évaluation du modèle
predictions = model.predict(X_test_scaled)
predictions = (predictions > 0.5).astype(int)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Log dans MLflow
mlflow.set_experiment("Meteo_cosin_rnn_xgb_ameliore")
with mlflow.start_run(run_name="rnn_xgb1_ameliore"):
    mlflow.log_params({"epochs": 100, "batch_size": 32, "layers": "3+dropout", "units": "64-32", "activation": "relu"})
    mlflow.log_metrics({"accuracy": history.history['val_accuracy'][-1]})
    mlflow.keras.log_model(model, "model_ameliore")

# Résultat obtenu :   
# precision    recall  f1-score   support
#
#           0       0.89      0.89      0.89     28213
#           1       0.89      0.89      0.89     28512
#
#    accuracy                           0.89     56725
#   macro avg       0.89      0.89      0.89     56725
#weighted avg       0.89      0.89      0.89     56725



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

# Résultat obtenu:#      precision    recall  f1-score   support
#
#           0       0.89      0.93      0.91     28213
#           1       0.92      0.89      0.91     28512
#
#    accuracy                           0.91     56725
#   macro avg       0.91      0.91      0.91     56725
#weighted avg       0.91      0.91      0.91     56725

'''
print("NNR 4 - ne fonctionne pas - "*10)
# Points clés de cette approche :
# Bidirectional LSTM : Utilisation de couches LSTM bidirectionnelles pour traiter les séquences d'entrée dans les deux directions, 
# ce qui peut améliorer la capture des dépendances à long terme.
# Dropout : Application de dropout pour réduire le surajustement.
# Early Stopping : Utilisation de l'arrêt précoce pour interrompre l'entraînement si la performance sur l'ensemble de validation 
# cesse de s'améliorer, afin d'éviter le surajustement.
# Ce modèle suppose que nos données peuvent être séquentiellement pertinentes, ce qui est typique pour les tâches de traitement de 
# texte ou de séries temporelles. Nos données sont de nature séquentielle et l'approche avec LSTM pourrait être la plus appropriée.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.tensorflow

# Chargement et préparation des données
df = pd.read_csv('df_full_cosin.csv')
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by='Date').drop(["Date", "Location"], axis=1)

# Encodage des labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['RainTomorrow'].values)

# Supposons que vos données sont déjà encodées en indices pour l'embedding
# et que 'data' représente ces indices
X = df.drop('RainTomorrow', axis=1).values
y = df['RainTomorrow'].values

# Calcul de input_dim comme étant le maximum + 1 parmi tous les indices dans X
input_dim = np.max(X) + 1

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Construction du modèle
model = Sequential([
    Embedding(input_dim=int(input_dim), output_dim=64, input_length=X_train.shape[1]),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dropout(0.25),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=1)

predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)
print(classification_report(y_test, predictions))

mlflow.tensorflow.autolog()
with mlflow.start_run(run_name="rnn_lstm_ameliore"):
    mlflow.log_params({
        "epochs": 100,
        "batch_size": 64,
        "layers": "LSTM Bidirectional",
        "units": "64-32",
        "activation": "sigmoid"
    })
    mlflow.log_metrics({
        "accuracy": max(history.history['val_accuracy']),
        "loss": min(history.history['val_loss'])
    })
'''