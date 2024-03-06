import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


# Chargement des données
chemin = "C:\\Users\\Utilisateur\\Mon Drive\\PROJET_METEO_DEC23_DS\\PRODUCTION\\Patrick\\data_meteo_patrick\\df_49_knn_4-clean_years.csv"
df = pd.read_csv(chemin, parse_dates=['Date'], index_col='Date')

# Encodage One-hot pour 'Location'
df = pd.get_dummies(df, columns=['Location'])

# Séparation des données en caractéristiques (X) et cible (y)
X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']

# Normalisation des caractéristiques numériques
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Préprocesseur qui inclut la normalisation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

# Application du préprocesseur
X_processed = preprocessor.fit_transform(X)

# Encodage One-hot pour y
y_processed = to_categorical(y)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

# Ajustement des dimensions pour LSTM
# Augmentation de la dimensionnalité pour le modèle LSTM

# Modèle LSTM pour la classification binaire
model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
    Dense(2, activation='softmax')  # 2 neurones de sortie car y est encodé en one-hot
])

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Ajustement des dimensions pour le modèle LSTM
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

# Construction du modèle LSTM
# Définissez une fonction pour créer un modèle LSTM avec une taille d'unités spécifique
def create_lstm_model(units):
    model = Sequential([
        LSTM(units, activation='relu', input_shape=(1, X_train.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(units, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Entraînement du modèle LSTM
model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
    Dense(2, activation='softmax')
])
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Évaluation du modèle LSTM
model_lstm.evaluate(X_test_lstm, y_test)

# Modèle Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, np.argmax(y_train, axis=1))  # np.argmax pour convertir y de one-hot à label

# Prédictions et évaluation du modèle Random Forest
y_pred_rf = model_rf.predict(X_test)
print("Accuracy du modèle Random Forest:", accuracy_score(np.argmax(y_test, axis=1), y_pred_rf))
print(classification_report(np.argmax(y_test, axis=1), y_pred_rf))


# Tuning des hyperparamètres pour Random Forest avec GridSearchCV
# Définition de la grille de recherche
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4, 6, 8, 10],
    'criterion' :['gini', 'entropy']
}

# Création du modèle de recherche sur grille
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Exécution de la recherche sur grille
grid_search.fit(X_train, y_train)

# Affichage des meilleurs paramètres et du score associé
print("Meilleurs paramètres : ", grid_search.best_params_)
print("Meilleure précision : ", grid_search.best_score_)


rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
grid_search.fit(X_train, y_train)

best_grid = grid_search.best_estimator_

# Utilisation du meilleur modèle pour des prédictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Évaluation du modèle avec les meilleurs hyperparamètres
print("Accuracy du meilleur modèle Random Forest:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Ensemble Learning avec Bagging
bagging_model = BaggingClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100),
    n_estimators=10,
    random_state=42,
)

bagging_model.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred = bagging_model.predict(X_test)

# Évaluation des performances
# Exactitude (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Autres métriques de classification
print(classification_report(y_test, y_pred))


# Feature Engineering : Ajout de nouvelles caractéristiques
df['temp_diff'] = df['MaxTemp'] - df['MinTemp']  # Différence de température
df['temp_scaled'] = (df['temp_diff'] - df['temp_diff'].mean()) / df['temp_diff'].std()  # Normalisation

# Entraînement et évaluation d'un modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))



#  Modèle perceptron multicouche (MLP) avec une seule couche cachée



# Créer les données d'entraînement factices
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [[0], [1], [1], [0]]

# Définir les paramètres du modèle
input_dim = len(X_train[0])
hidden_units = 10
output_dim = 1

# Créer les placeholders pour les données d'entrée et de sortie
inputs = tf.placeholder(tf.float32, shape=[None, input_dim])
labels = tf.placeholder(tf.float32, shape=[None, output_dim])

# Créer les poids et les biais du réseau
weights_hidden = tf.Variable(tf.random.normal([input_dim, hidden_units]))
bias_hidden = tf.Variable(tf.zeros([hidden_units]))
weights_output = tf.Variable(tf.random.normal([hidden_units, output_dim]))
bias_output = tf.Variable(tf.zeros([output_dim]))

# Définir les opérations du réseau de neurones
hidden_layer = tf.nn.sigmoid(tf.matmul(inputs, weights_hidden) + bias_hidden)
output_layer = tf.matmul(hidden_layer, weights_output) + bias_output

# Définir la fonction de perte et l'optimiseur
loss = tf.reduce_mean(tf.square(output_layer - labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)

# Entraîner le modèle
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
        _, current_loss = sess.run([train_op, loss], feed_dict={inputs: X_train, labels: y_train})
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss}")

    # Prédire avec le modèle entraîné
    predictions = sess.run(output_layer, feed_dict={inputs: X_train})
    print("Predictions:", predictions)


