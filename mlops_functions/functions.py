import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fanalysis.mca import MCA
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import scipy


#Fonction de chargement

def load_data(file_path):
    return pd.read_csv(file_path)

#Fonction filtre et suppression

def filter_summer_season(df):
    return df[df['Season'] == 'Summer']

def filter_by_year(df, year_threshold=2000):
    return df[df['Year'] >= year_threshold]

def drop_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop, axis=1)

def display_missing_values(df):
    # Calcul des valeurs manquantes par variable
    missing_values_per_variable = df.isnull().sum()

    # Affichage du résultat
    print("Valeurs manquantes par variable :")
    print(missing_values_per_variable)


def drop_missing_values(df, columns_to_check):
    # Supprimer les lignes avec des valeurs manquantes dans les colonnes spécifiées
    df_net = df.dropna(subset=columns_to_check)
    
    # Afficher les valeurs manquantes après la suppression
    print("Valeurs manquantes après suppression :")
    print(df_net.isnull().sum())

    return df_net



#Fonction de regroupement

def create_age_classes(df):
    # Intervalles d'âge
    intervalles = [0, 21, 25, 33, float('inf')]
    labels = ['< 21 ans', '21 - 24 ans', '25-32 ans', '> 32 ans']
    
    # Nouvelle colonne "Classe_age" basée sur les intervalles définis
    df['Classe_age'] = pd.cut(df['Age'], bins=intervalles, labels=labels, right=False)
    
    # Convertir la colonne "Classe_age" en catégorie si nécessaire
    df['Classe_age'] = df['Classe_age'].astype('category')
    
    return df[['Age', 'Classe_age']]

def create_height_classes(df):
    # Intervalles de taille
    intervalles_taille = [float('-inf'), 165, 173, 186, float('inf')]
    labels_taille = ['< 165 cm', '165 - 172 cm', '173 - 185 cm', '> 185 cm']
    
    # Nouvelle colonne "Classe_height" basée sur les intervalles définis
    df['Classe_height'] = pd.cut(df['Height'], bins=intervalles_taille, labels=labels_taille, right=False)
    
    # Convertir la colonne "Classe_height" en catégorie si nécessaire
    df['Classe_height'] = df['Classe_height'].astype('category')
    
    return df[['Height', 'Classe_height']]

def create_weight_classes(df):
    # Intervalles de poids
    intervalles_poids = [float('-inf'), 65, 74, 81, float('inf')]
    labels_poids = ['< 65 kg', '65 - 73 kg', '74 - 80 kg', '> 80 kg']
    
    # Nouvelle colonne "Classe_weight" basée sur les intervalles définis
    df['Classe_weight'] = pd.cut(df['Weight'], bins=intervalles_poids, labels=labels_poids, right=False)
    
    # Convertir la colonne "Classe_weight" en catégorie si nécessaire
    df['Classe_weight'] = df['Classe_weight'].astype('category')
    
    return df[['Weight', 'Classe_weight']]


