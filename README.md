```markdown
# README: Analyse et Modélisation de la Satisfaction Client dans le Transport

Ce notebook vise à prédire le niveau de satisfaction client (`satisfaction_client`) pour une entreprise de transport, en utilisant diverses caractéristiques des trajets. Le processus inclut le chargement des données, le nettoyage, la préparation, l'entraînement de plusieurs modèles de classification, l'optimisation des hyperparamètres et l'évaluation des performances.

## Table des Matières
1. [Chargement et Exploration Initiale des Données](#1-chargement-et-exploration-initiale-des-données)
2. [Nettoyage des Données](#2-nettoyage-des-données)
3. [Préparation des Données pour la Modélisation](#3-préparation-des-données-pour-la-modélisation)
4. [Définition des Modèles et des Pipelines](#4-définition-des-modèles-et-des-pipelines)
5. [Optimisation des Hyperparamètres (Première Itération)](#5-optimisation-des-hyperparamètres-première-itération)
6. [Évaluation des Modèles (Première Itération)](#6-évaluation-des-modèles-première-itération)
7. [Analyse de l'Déséquilibre des Classes](#7-analyse-de-l'déséquilibre-des-classes)
8. [Ré-optimisation avec Gestion du Déséquilibre des Classes (SMOTE)](#8-ré-optimisation-avec-gestion-du-déséquilibre-des-classes-smote)
9. [Évaluation Finale des Modèles](#9-évaluation-finale-des-modèles)

---

### 1. Chargement et Exploration Initiale des Données
Le notebook commence par charger le fichier `dataset_planification_transport_LK.csv` dans un DataFrame pandas. Une première exploration est effectuée pour comprendre la structure des données, identifier les types de colonnes et détecter les valeurs manquantes.

- **Chargement du CSV** dans `df_planification`.
- **`df_planification.head()`**: Affichage des premières lignes.
- **`df_planification.info()`**: Informations générales sur le DataFrame.
- **`df_planification.isnull().sum()`**: Comptage des valeurs manquantes par colonne.
- **`df_planification.describe()`**: Statistiques descriptives pour les colonnes numériques.
- Identification des colonnes catégorielles.

### 2. Nettoyage des Données
Étant donné la présence de nombreuses valeurs manquantes, les lignes contenant au moins une valeur manquante sont supprimées pour créer un DataFrame propre `df_planification_cleaned`.

- **Suppression des lignes avec valeurs manquantes** (`dropna`).
- Vérification de l'absence de valeurs manquantes après le nettoyage.

### 3. Préparation des Données pour la Modélisation
La variable cible (`satisfaction_client`) est séparée des caractéristiques. Les données sont ensuite divisées en ensembles d'entraînement et de test.

- **Définition de `y` (variable cible)**: `satisfaction_client`.
- **Définition de `X` (caractéristiques)**: Toutes les colonnes sauf `trajet_id` et `satisfaction_client`.
- **Division des données** en `X_train`, `X_test`, `y_train`, `y_test` avec `train_test_split` (ratio de test de 20%, `random_state=42`).

### 4. Définition des Modèles et des Pipelines
Trois modèles de classification (`LogisticRegression`, `RandomForestClassifier`, `SVC`) sont définis. Des pipelines sont créés pour inclure un préprocesseur (`ColumnTransformer`) qui gère la standardisation des colonnes numériques (`StandardScaler`) et l'encodage One-Hot des colonnes catégorielles (`OneHotEncoder`).

- **Importation des librairies nécessaires**: `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`, `SVC`.
- **Identification des colonnes numériques et catégorielles**.
- **Création du `preprocessor`**.
- **Définition des modèles**.
- **Construction des pipelines** intégrant le préprocesseur et chaque classifieur.

### 5. Optimisation des Hyperparamètres (Première Itération)
Une grille d'hyperparamètres (`param_grid`) est définie pour chaque modèle. `GridSearchCV` est utilisé pour trouver les meilleurs hyperparamètres pour chaque modèle en utilisant la validation croisée.

- **`param_grid`**: Définition des plages de valeurs pour les hyperparamètres de chaque modèle.
- **`GridSearchCV`**: Exécution de la recherche sur grille pour chaque pipeline.
- **Stockage des meilleurs estimateurs, scores et paramètres** dans `best_models`.
- **Identification du meilleur modèle global** basé sur le score de validation croisée.

### 6. Évaluation des Modèles (Première Itération)
Le meilleur modèle identifié lors de la première itération de `GridSearchCV` est évalué sur l'ensemble de test en utilisant le rapport de classification et la matrice de confusion.

- **Rapport de classification (`classification_report`)**:
  - **Précision, Rappel, F1-score**: Très faibles pour la plupart des classes (0.00).
  - **Accuracy**: Environ 0.23.
- **Matrice de confusion (`confusion_matrix`)**: Indique que le modèle prédit majoritairement la classe 3, ignorant les autres classes. Ceci est un signe clair de **déséquilibre des classes**.

### 7. Analyse de l'Déséquilibre des Classes
La distribution de la variable cible `satisfaction_client` dans l'ensemble d'entraînement est visualisée et quantifiée, confirmant un déséquilibre entre les classes.

- **Visualisation de la distribution** avec `seaborn.countplot`.
- **Distribution numérique** (`value_counts(normalize=True)`) montrant que la classe 3 est la plus fréquente (environ 23.7%), tandis que la classe 5 est la moins fréquente (environ 16.9%).

### 8. Ré-optimisation avec Gestion du Déséquilibre des Classes (SMOTE)
Pour adresser le déséquilibre des classes, la technique de sur-échantillonnage **SMOTE (Synthetic Minority Over-sampling Technique)** est intégrée dans les pipelines. Les grilles d'hyperparamètres sont également mises à jour pour inclure l'option `class_weight='balanced'` lorsque disponible.

- **Importation de `SMOTE` depuis `imblearn.over_sampling`**.
- **Création de `ImbPipeline`** (pipeline compatible avec `imblearn`) incluant `preprocessor`, `smote` et le classifieur.
- **Mise à jour de `param_grid`** pour inclure `classifier__class_weight`.
- **Nouvelle exécution de `GridSearchCV`** avec les pipelines et les grilles d'hyperparamètres mis à jour.
- **Identification du meilleur modèle global** après cette deuxième itération.
  - Le meilleur modèle est `SVC` avec un score de validation croisée de 0.2071.

### 9. Évaluation Finale des Modèles
Le meilleur modèle (SVC) de la deuxième itération (avec SMOTE) est évalué sur l'ensemble de test.

- **Rapport de classification (`classification_report`)**:
  - **Précision, Rappel, F1-score**: Les scores sont toujours bas, mais il y a une certaine amélioration dans la prédiction de toutes les classes, bien qu'aucune classe ne soit prédite avec une grande confiance.
  - **Accuracy**: Environ 0.21. L'accuracy a légèrement diminué par rapport à la première itération, mais les métriques par classe sont plus équilibrées (même si faibles).
- **Matrice de confusion (`confusion_matrix`)**: Montre que le modèle essaie maintenant de prédire toutes les classes, mais avec beaucoup de misclassifications. La classe 3 n'est plus la seule prédite.

## Conclusion et Perspectives
Les performances des modèles restent faibles, même après l'application de techniques de gestion du déséquilibre des classes comme SMOTE. Cela suggère que les caractéristiques actuelles ne sont peut-être pas suffisamment informatives pour prédire la satisfaction client avec une grande précision. Des pistes d'amélioration pourraient inclure :

- **Ingénierie de caractéristiques (Feature Engineering)**: Créer de nouvelles caractéristiques à partir des données existantes ou intégrer des sources de données supplémentaires.
- **Exploration de modèles plus complexes**: Essayer des modèles d'apprentissage profond ou des ensembles de modèles.
- **Collecte de données supplémentaires**: Plus de données ou des données plus pertinentes pourraient améliorer la prédiction.
- **Analyse d'erreurs**: Examiner en détail les erreurs de prédiction pour comprendre où les modèles échouent.

Le modèle actuel ne serait pas adapté à une mise en production en l'état en raison de ses faibles performances.
```
