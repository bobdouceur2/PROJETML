# Importation des bibliothèques nécessaires pour la manipulation des tableaux et des données
import numpy as np  # Bibliothèque pour les opérations sur les tableaux et les calculs numériques
import pandas as pd  # Bibliothèque pour la manipulation et l'analyse des données structurées
import streamlit as st  # Bibliothèque pour créer des applications web interactives

# Importation des modules de scikit-learn pour la préparation des données et les modèles
from sklearn.model_selection import train_test_split, GridSearchCV  # Pour diviser les données et effectuer des recherches sur les hyperparamètres
from sklearn.feature_extraction.text import TfidfVectorizer  # Pour convertir le texte en vecteurs TF-IDF

# Importation des bibliothèques TensorFlow et TensorFlow Datasets
import tensorflow_datasets as tfds  # Pour charger des jeux de données depuis TensorFlow Datasets
import tensorflow as tf  # Pour les opérations de machine learning et les réseaux de neurones

# Importation des couches et des modèles de Keras
from tensorflow.keras.layers import Dense, Dropout, Input  # Pour définir les couches du modèle de réseau de neurones
from tensorflow.keras import Sequential  # Pour créer des modèles séquentiels avec Keras

# Importation de la classe KerasClassifier de scikeras
from scikeras.wrappers import KerasClassifier  # Pour intégrer les modèles Keras dans scikit-learn

# Importation de matplotlib pour le traçage des graphiques
import matplotlib.pyplot as plt  # Pour créer des graphiques et des visualisations

# Importation de la bibliothèque time pour mesurer le temps d'exécution
import time  # Pour mesurer et afficher le temps d'exécution des opérations

# Début du chronométrage
start_time = time.time()

# Télécharger les données IMDB Movie Reviews de tfds
tfds_dataset = tfds.load('imdb_reviews', as_supervised=True)  # Charger les critiques de films IMDB
tfds_train_dataset, tfds_test_dataset = tfds_dataset['train'], tfds_dataset['test']  # Séparer en ensembles d'entraînement et de test

# Initialisation des listes pour stocker les critiques et les étiquettes
reviews = []
labels = []
num_reviews_to_extract = 1000  # Nombre de critiques à extraire
count = 0  # Compteur pour le nombre de critiques extraites

# Compteurs pour le nombre de critiques positives et négatives
num_positive_reviews = 0
num_negative_reviews = 0

# Boucle pour extraire les critiques et les étiquettes de l'ensemble d'entraînement
for example, label in tfds_train_dataset:
    if count >= num_reviews_to_extract:
        break  # Arrêter l'extraction après avoir atteint le nombre souhaité
    if label == 1:
        num_positive_reviews += 1  # Incrémenter le compteur de critiques positives
    else:
        num_negative_reviews += 1  # Incrémenter le compteur de critiques négatives
    reviews.append(example.numpy())  # Ajouter la critique à la liste
    labels.append(label.numpy())  # Ajouter l'étiquette à la liste
    count += 1  # Incrémenter le compteur

# Affichage du nombre de critiques positives et négatives
print("Nombre de critiques positives :", num_positive_reviews)
print("Nombre de critiques négatives :", num_negative_reviews)

# Extraction des caractéristiques (TF-IDF) à partir des critiques
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(reviews)  # Conversion des critiques en vecteurs TF-IDF
y = np.array(labels)  # Conversion des étiquettes en tableau NumPy

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Transformation des matrices en tableaux numpy si nécessaire
if not isinstance(X_train, np.ndarray):
    X_train = X_train.toarray()  # Conversion en tableau NumPy
if not isinstance(X_test, np.ndarray):
    X_test = X_test.toarray()  # Conversion en tableau NumPy

# Définir une fonction pour créer le modèle de réseau de neurones
def create_model(optimizer='adam', neurons1=128, neurons2=128, dropout_rate=0.5):
    model = Sequential()  # Initialiser un modèle séquentiel
    model.add(Input(shape=(1000,)))  # Ajouter une couche d'entrée avec la forme appropriée
    model.add(Dense(neurons1, activation='relu'))  # Ajouter une couche dense avec activation ReLU
    model.add(Dropout(dropout_rate))  # Ajouter une couche de dropout pour éviter le surapprentissage
    model.add(Dense(neurons2, activation='relu'))  # Ajouter une deuxième couche dense
    model.add(Dropout(dropout_rate))  # Ajouter une deuxième couche de dropout
    model.add(Dense(2, activation='softmax'))  # Ajouter une couche de sortie avec activation softmax
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  # Compiler le modèle
    return model

# Wrapper du modèle Keras dans un KerasClassifier pour scikit-learn
model = KerasClassifier(model=create_model, verbose=0)

# Définir la grille de recherche des hyperparamètres
param_grid = {
    'batch_size': [16, 32],  # Taille des lots
    'epochs': [10, 15, 20, 30],  # Nombre d'époques
    'optimizer': ['adam'],  # Optimiseur
    'model__neurons1': [128],  # Nombre de neurones dans la première couche dense
    'model__neurons2': [128],  # Nombre de neurones dans la deuxième couche dense
    'model__dropout_rate': [0.7, 0.9]  # Taux de dropout
}

# Utiliser GridSearchCV pour rechercher les meilleurs hyperparamètres
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, return_train_score=True)
grid_result = grid.fit(X_train, tf.keras.utils.to_categorical(y_train))  # Adapter GridSearchCV aux données

# Afficher les meilleurs hyperparamètres trouvés par GridSearchCV
print("Meilleurs hyperparamètres : %s avec une accuracy de %f" % (grid_result.best_params_, grid_result.best_score_))

# Entraîner le modèle avec les meilleurs hyperparamètres et enregistrer l'historique d'entraînement
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Définir le callback d'arrêt anticipé
best_model = grid_result.best_estimator_  # Obtenir le meilleur modèle de l'estimation de la grille
history = best_model.fit(X_train, tf.keras.utils.to_categorical(y_train),
                         validation_data=(X_test, tf.keras.utils.to_categorical(y_test)),
                         callbacks=[early_stopping])  # Entraîner le modèle

# Évaluer le modèle sur l'ensemble de test
accuracy = best_model.score(X_test, tf.keras.utils.to_categorical(y_test))  # Évaluer le modèle sur les données de test
print("Accuracy sur l'ensemble de test : %.2f%%" % (accuracy * 100))  # Afficher l'exactitude

# Utiliser Streamlit pour créer l'interface utilisateur
st.title("UltraOptimisation des Hyperparamètres 3 couches de neurones pour DNN avec IMDB")

# Affichage de la performance du modèle
st.write(f"Meilleurs hyperparamètres : {grid_result.best_params_}")  # Afficher les meilleurs hyperparamètres
st.write(f"Accuracy sur l'ensemble de test : {accuracy:.4f}")  # Afficher l'exactitude sur le test

# Afficher les graphiques de la perte et de la précision au fil des époques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # Créer une figure avec deux sous-graphiques

# Graphique de la perte (loss) par rapport aux époques
ax1.plot(history.history_['loss'], label='Training Loss')  # Tracer la perte d'entraînement
ax1.plot(history.history_['val_loss'], label='Validation Loss')  # Tracer la perte de validation
ax1.set_title('Loss par Epoch')  # Titre du graphique de la perte
ax1.set_xlabel('Epochs')  # Étiquette de l'axe des x
ax1.set_ylabel('Loss')  # Étiquette de l'axe des y
ax1.legend()  # Afficher la légende

# Graphique de la précision (accuracy) par rapport aux époques
ax2.plot(history.history_['accuracy'], label='Training Accuracy')  # Tracer la précision d'entraînement
ax2.plot(history.history_['val_accuracy'], label='Validation Accuracy')  # Tracer la précision de validation
ax2.set_title('Accuracy par Epoch')  # Titre du graphique de la précision
ax2.set_xlabel('Epochs')  # Étiquette de l'axe des x
ax2.set_ylabel('Accuracy')  # Étiquette de l'axe des y
ax2.legend()  # Afficher la légende

# Afficher les graphiques dans Streamlit
st.pyplot(fig)

# Afficher les résultats des performances pour les différents paramètres
results = pd.DataFrame(grid_result.cv_results_)  # Créer un DataFrame avec les résultats de la recherche en grille
results = results[['param_batch_size', 'param_epochs', 'param_optimizer', 'param_model__neurons1', 'param_model__neurons2', 'param_model__dropout_rate', 'mean_test_score', 'mean_train_score']]  # Sélectionner les colonnes pertinentes
st.write("Performance par Hyperparamètres")  # Titre de la section des performances
st.write(results)  # Afficher les résultats dans Streamlit

# Fin du chronométrage
end_time = time.time()
# Calcul du temps total pris
total_time = end_time - start_time
print("Total Time Taken: {:.2f} seconds".format(total_time))  # Afficher le temps total pris
