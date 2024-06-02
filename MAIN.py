# Importation des bibliothèques nécessaires
import numpy as np  # Pour la manipulation des tableaux et les calculs numériques
import pandas as pd  # Pour la manipulation et l'analyse des données
import streamlit as st  # Pour créer l'application web interactive

# Importation des modules de scikit-learn pour la préparation des données et les modèles
from sklearn.model_selection import train_test_split  # Pour diviser les données en ensembles d'entraînement et de test
from sklearn.feature_extraction.text import TfidfVectorizer  # Pour convertir le texte en vecteurs TF-IDF
from sklearn.linear_model import LogisticRegression  # Pour le modèle de régression logistique
from sklearn.tree import DecisionTreeClassifier  # Pour le modèle d'arbre de décision
from sklearn.ensemble import RandomForestClassifier  # Pour le modèle de forêt aléatoire
from sklearn.neighbors import KNeighborsClassifier  # Pour le modèle des k plus proches voisins
from sklearn.neural_network import MLPClassifier  # Pour le modèle de perceptron multicouche
from sklearn.datasets import fetch_openml  # Pour importer des jeux de données à partir d'OpenML
from sklearn.metrics import accuracy_score  # Pour calculer l'exactitude des modèles

# Importation des bibliothèques supplémentaires nécessaires
import nltk  # Pour le traitement du langage naturel
import tensorflow_datasets as tfds  # Pour charger des jeux de données de TensorFlow
from nltk.corpus import movie_reviews  # Pour accéder aux critiques de films dans NLTK
import requests  # Pour faire des requêtes HTTP
import tensorflow as tf  # Pour les opérations de machine learning et les réseaux de neurones
from tensorflow.keras import layers  # Pour créer des couches de réseau de neurones
from tensorflow.keras.layers import Dense, Flatten, Input  # Pour les couches spécifiques de Keras
from tensorflow.keras import Sequential  # Pour créer un modèle séquentiel Keras
import matplotlib.pyplot as plt  # Pour tracer des graphiques

# Titre de l'application Streamlit
st.title("Comparateur de Classificateurs avec IMDB")

# Télécharger les données IMDB Movie Reviews de TensorFlow datasets
tfds_dataset = tfds.load('imdb_reviews', as_supervised=True)
tfds_train_dataset, tfds_test_dataset = tfds_dataset['train'], tfds_dataset['test']

# Initialisation des listes pour stocker les critiques et les étiquettes
reviews = []
labels = []
num_reviews_to_extract = 10000  # Nombre de critiques à extraire
count = 0

# Compteurs pour le nombre de critiques positives et négatives
num_positive_reviews = 0
num_negative_reviews = 0

# Extraction des critiques et des étiquettes
for example, label in tfds_train_dataset:
    if count >= num_reviews_to_extract:
        break  # Stopper l'extraction après avoir atteint le nombre souhaité
    if label == 1:
        num_positive_reviews += 1  # Compter les critiques positives
    else:
        num_negative_reviews += 1  # Compter les critiques négatives
    reviews.append(example.numpy())  # Ajouter la critique à la liste
    labels.append(label.numpy())  # Ajouter l'étiquette à la liste
    count += 1

# Affichage du nombre de critiques positives et négatives
print("Nombre de critiques positives :", num_positive_reviews)
print("Nombre de critiques négatives :", num_negative_reviews)

# Extraction des caractéristiques (TF-IDF) à partir des critiques
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(reviews)
y = np.array(labels)

# Instanciation des classificateurs
logistic_reg = LogisticRegression(C=1, max_iter=500, solver='liblinear')  # Régression logistique
decision_tree_clf = DecisionTreeClassifier()  # Arbre de décision
random_forest_clf = RandomForestClassifier()  # Forêt aléatoire
knn_clf = KNeighborsClassifier()  # K plus proches voisins
mlp_clf = MLPClassifier()  # Perceptron multicouche

# Réseau de neurones profond (DNN) avec Keras
dnn_clf = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(1000,)),  # Couche dense avec 128 neurones et activation ReLU
    tf.keras.layers.Dropout(0.9),  # Couche de dropout pour éviter le surapprentissage
    tf.keras.layers.Dense(128, activation='relu'),  # Deuxième couche dense
    tf.keras.layers.Dropout(0.9),  # Deuxième couche de dropout
    tf.keras.layers.Dense(2, activation='softmax')  # Couche de sortie avec activation softmax
])

# Compilation du modèle DNN
dnn_clf.compile(optimizer='adam',
                loss='categorical_crossentropy',  # Utiliser 'categorical_crossentropy' avec des étiquettes one-hot
                metrics=['accuracy'])

# Liste des classificateurs avec leurs noms
classifiers = [
    ('Logistic Regression', logistic_reg),
    ('Decision Tree', decision_tree_clf),
    ('Random Forest', random_forest_clf),
    ('K-Nearest Neighbors', knn_clf),
    ('MLP', mlp_clf),
    ('DNN', dnn_clf)
]

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Conversion des données en tableaux NumPy si nécessaire
if not isinstance(X_train, np.ndarray):
    X_train = X_train.toarray()
if not isinstance(X_test, np.ndarray):
    X_test = X_test.toarray()

# Affichage de la performance comparée des classificateurs
results = []
for name, classifier in classifiers:
    if name == 'DNN':
        classifier.fit(X_train, tf.keras.utils.to_categorical(y_train), epochs=10, batch_size=16)  # Entraînement du DNN
        loss, accuracy = classifier.evaluate(X_test, tf.keras.utils.to_categorical(y_test))  # Évaluation du DNN
        results.append((name, accuracy))
    else:
        classifier.fit(X_train, y_train)  # Entraînement des autres classificateurs
        y_pred = classifier.predict(X_test)  # Prédiction des étiquettes sur l'ensemble de test
        accuracy = accuracy_score(y_test, y_pred)  # Calcul de l'exactitude
        results.append((name, accuracy))

# Création d'un DataFrame pour afficher les résultats
results_df = pd.DataFrame(results, columns=['Classificateur', 'Accuracy'])
st.write(results_df)

# Obtenir les noms des 3 meilleurs classificateurs selon l'exactitude
top_classifiers = results_df.nlargest(3, 'Accuracy')['Classificateur'].values

# Sous-titre pour la section de prédiction personnalisée
st.subheader("Prédiction d'un critique personnalisée")

# Sélection du classificateur par l'utilisateur
selected_classifier = st.selectbox("Sélectionnez un Classificateur", [name for name, _ in classifiers])

# Entraînement et évaluation du modèle sélectionné
for name, classifier in classifiers:
    if name == selected_classifier:
        st.subheader(f"Évaluation de {name}")
        if name == 'DNN':
            loss, accuracy = classifier.evaluate(X_test, tf.keras.utils.to_categorical(y_test))
        else:
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.4f}")

# L'utilisateur peut entrer un avis de film
avis = st.text_area("Entrez un avis de film en anglais pour tester", "The movie was bad. The animation and the graphics were poor. I would not recommend this movie.")
example = vectorizer.transform([avis])
if not isinstance(example, np.ndarray):
    example = example.toarray()

# Prédiction selon le classificateur choisi
if selected_classifier == 'DNN':
    if (np.argmax(dnn_clf.predict(example)) == 1):
        st.write('Avis positif')
    else:
        st.write('Avis négatif')
else:
    for name, classifier in classifiers:
        if name == selected_classifier:
            prediction = classifier.predict(example)
            st.write('Avis positif' if prediction == 1 else 'Avis négatif')
