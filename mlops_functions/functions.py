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



def transform_base_ACM(df):
    df['Sport'].replace({
        'Judo': 'Combat',
        'Wrestling': 'Combat',
        'Taekwondo': 'Combat',
        'Fencing': 'Combat',
        'Boxing': 'Combat',
        'Badminton': 'Raquette',
        'Tennis': 'Raquette',
        'Table Tennis': 'Raquette',
        'Swimming': 'Natation',
        'Synchronized Swimming': 'Natation',
        'Basketball': 'Sport collectif',
        'Handball': 'Sport collectif',
        'Football': 'Sport collectif',
        'Hockey': 'Sport collectif',
        'Water Polo': 'Sport collectif',
        'Softball': 'Sport collectif',
        'Volleyball': 'Sport collectif',
        'Baseball': 'Sport collectif',
        'Rugby Sevens': 'Sport collectif',
        'Beach Volleyball': 'Sport collectif',
        'Athletics': 'Athlétisme',
        'Modern Pentathlon': 'Athlétisme',
        'Triathlon': 'Athlétisme',
        'Gymnastics': 'Gymnastique',
        'Rhythmic Gymnastics': 'Gymnastique',
        'Trampolining': 'Gymnastique',
        'Sailing': "Sur l'eau",
        'Rowing': "Sur l'eau",
        'Diving': "Sur l'eau",
        'Canoeing': "Sur l'eau",
        'Weightlifting': 'Autres sports',
        'Cycling': 'Autres sports',
        'Equestrianism': 'Autres sports',
        'Archery': 'Autres sports',
        'Shooting': 'Autres sports',
        'Golf': 'Autres sports'
    }, inplace=True)

    df['NOC'].replace({
    'CHN': 'Asie',
    'FIN': 'Europe',
    'ROU': 'Europe',
    'NOR': 'Europe',
    'NED': 'Europe',
    'FRA': 'Europe',
    'EST': 'Europe',
    'ESP': 'Europe',
    'EGY': 'Afrique',
    'ITA': 'Europe',
    'AZE': 'Asie',
    'RUS': 'Europe',
    'ARG': 'Amérique du Sud',
    'CUB': 'Amérique du Nord',
    'BLR': 'Europe',
    'GRE': 'Europe',
    'CMR': 'Afrique',
    'MEX': 'Amérique du Nord',
    'USA': 'Amérique du Nord',
    'NCA': 'Amérique centrale',
    'ALG': 'Afrique',
    'BRN': 'Asie',
    'IRQ': 'Asie',
    'QAT': 'Asie',
    'PAK': 'Asie',
    'IRI': 'Asie',
    'CAN': 'Amérique du Nord',
    'IRL': 'Europe',
    'AUS': 'Océanie',
    'RSA': 'Afrique',
    'MAR': 'Afrique',
    'ERI': 'Afrique',
    'SUD': 'Afrique',
    'BEL': 'Europe',
    'KAZ': 'Asie',
    'BRU': 'Asie',
    'KUW': 'Asie',
    'MAS': 'Asie',
    'INA': 'Asie',
    'UZB': 'Asie',
    'UAE': 'Asie',
    'KGZ': 'Asie',
    'TJK': 'Asie',
    'JPN': 'Asie',
    'GER': 'Europe',
    'ETH': 'Afrique',
    'TUR': 'Asie',
    'SRI': 'Asie',
    'ARM': 'Asie',
    'CIV': 'Afrique',
    'KEN': 'Afrique',
    'NGR': 'Afrique',
    'BRA': 'Amérique du Sud',
    'SYR': 'Asie',
    'CHI': 'Amérique du Sud',
    'SUI': 'Europe',
    'SWE': 'Europe',
    'GUY': 'Amérique du Sud',
    'GEO': 'Asie',
    'POR': 'Europe',
    'ANG': 'Afrique',
    'COL': 'Amérique du Sud',
    'DJI': 'Afrique',
    'BAN': 'Asie',
    'JOR': 'Asie',
    'PLE': 'Asie',
    'SOM': 'Afrique',
    'KSA': 'Asie',
    'VEN': 'Amérique du Sud',
    'IND': 'Asie',
    'GBR': 'Europe',
    'GHA': 'Afrique',
    'UGA': 'Afrique',
    'TUN': 'Afrique',
    'SLO': 'Europe',
    'HON': 'Amérique centrale',
    'TKM': 'Asie',
    'MRI': 'Afrique',
    'POL': 'Europe',
    'NIG': 'Afrique',
    'SKN': 'Amérique du Nord',
    'NZL': 'Océanie',
    'LBR': 'Afrique',
    'SUR': 'Amérique du Sud',
    'NEP': 'Asie',
    'LBA': 'Afrique',
    'MGL': 'Asie',
    'PLW': 'Océanie',
    'LTU': 'Europe',
    'NAM': 'Afrique',
    'UKR': 'Europe',
    'ASA': 'Océanie',
    'PUR': 'Amérique du Nord',
    'SAM': 'Océanie',
    'RWA': 'Afrique',
    'CRO': 'Europe',
    'DMA': 'Amérique du Nord',
    'DEN': 'Europe',
    'MLT': 'Europe',
    'AUT': 'Europe',
    'SEY': 'Afrique',
    'DOM': 'Amérique du Nord',
    'BIZ': 'Amérique centrale',
    'PAR': 'Amérique du Sud',
    'URU': 'Amérique du Sud',
    'COM': 'Afrique',
    'MDV': 'Asie',
    'BEN': 'Afrique',
    'TTO': 'Amérique du Nord',
    'SGP': 'Asie',
    'PER': 'Amérique du Sud',
    'BER': 'Amérique du Nord',
    'SCG': 'Europe',
    'HUN': 'Europe',
    'CYP': 'Europe',
    'YEM': 'Asie',
    'LIB': 'Afrique',
    'OMA': 'Asie',
    'IOA': 'Océanie',
    'FIJ': 'Océanie',
    'VAN': 'Océanie',
    'JAM': 'Amérique du Nord',
    'MDA': 'Europe',
    'GUA': 'Amérique centrale',
    'BUL': 'Europe',
    'LAT': 'Europe',
    'SRB': 'Europe',
    'IVB': 'Amérique du Nord',
    'VIN': 'Amérique centrale',
    'ISL': 'Europe',
    'CRC': 'Amérique centrale',
    'ESA': 'Amérique centrale',
    'CAF': 'Afrique',
    'MAD': 'Afrique',
    'CHA': 'Afrique',
    'BIH': 'Europe',
    'GUM': 'Océanie',
    'PHI': 'Asie',
    'CAY': 'Amérique du Nord',
    'SVK': 'Europe',
    'BAR': 'Amérique du Nord',
    'ECU': 'Amérique du Sud',
    'PAN': 'Amérique centrale',
    'TLS': 'Asie',
    'GAB': 'Afrique',
    'BAH': 'Amérique du Nord',
    'SMR': 'Europe',
    'ISR': 'Asie',
    'THA': 'Asie',
    'BOT': 'Afrique',
    'ROT': 'Océanie',
    'KOR': 'Asie',
    'PRK': 'Asie',
    'MOZ': 'Afrique',
    'CPV': 'Afrique',
    'CZE': 'Europe',
    'LAO': 'Asie',
    'LUX': 'Europe',
    'AND': 'Europe',
    'ZIM': 'Afrique',
    'GRN': 'Amérique du Nord',
    'HKG': 'Asie',
    'LCA': 'Amérique du Nord',
    'HAI': 'Amérique du Nord',
    'FSM': 'Océanie',
    'MYA': 'Asie',
    'AFG': 'Asie',
    'SEN': 'Afrique',
    'MTN': 'Afrique',
    'COD': 'Afrique',
    'GUI': 'Afrique',
    'ANT': 'Amérique du Nord',
    'CGO': 'Afrique',
    'MKD': 'Europe',
    'BOL': 'Amérique du Sud',
    'TOG': 'Afrique',
    'SLE': 'Afrique',
    'MON': 'Europe',
    'GEQ': 'Afrique',
    'MNE': 'Europe',
    'ISV': 'Amérique du Nord',
    'PNG': 'Océanie',
    'TAN': 'Afrique',
    'COK': 'Océanie',
    'ALB': 'Europe',
    'MLI': 'Afrique',
    'SWZ': 'Afrique',
    'BDI': 'Afrique',
    'ARU': 'Amérique du Sud',
    'STP': 'Afrique',
    'NRU': 'Océanie',
    'GBS': 'Afrique',
    'ZAM': 'Afrique',
    'TPE': 'Asie',
    'CAM': 'Amérique centrale',
    'MAW': 'Afrique',
    'BHU': 'Asie',
    'VIE': 'Asie',
    'GAM': 'Afrique',
    'MHL': 'Océanie',
    'AHO': 'Océanie',
    'KIR': 'Océanie',
    'TUV': 'Océanie',
    'TGA': 'Océanie',
    'LIE': 'Europe',
    'KOS': 'Europe',
    'SOL': 'Océanie',
    'SSD': 'Afrique',
    'LES': 'Afrique',
    'BUR': 'Afrique',
        }, inplace=True)

    # Suppression des variables inutiles pour la suite de l'analyse
    df = df.drop(['ID', 'Name', 'Games', 'Year', 'City', 'Event'], axis=1)

    return df


#Fonction de v de cramer 

def calculate_cramer_v(df, categorical_vars):
    # Créez une table de contingence pour chaque paire de variables
    contingency_tables = {}
    for var1 in categorical_vars:
        for var2 in categorical_vars:
            contingency_table = pd.crosstab(df[var1], df[var2])
            contingency_tables[(var1, var2)] = contingency_table

    # Calculez le V de Cramer pour chaque paire de variables
    cramer_v_values = {}
    for (var1, var2), contingency_table in contingency_tables.items():
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        cramer_v_values[(var1, var2)] = cramers_v

    # Créez un DataFrame pour stocker les valeurs du V de Cramer
    cramer_df = pd.DataFrame(index=categorical_vars, columns=categorical_vars)
    for var1 in categorical_vars:
        for var2 in categorical_vars:
            cramer_df.loc[var1, var2] = cramer_v_values.get((var1, var2), cramer_v_values.get((var2, var1)))

    # Convertissez les valeurs en nombres décimaux
    cramer_df = cramer_df.astype(float)

    return cramer_df


def plot_cramer_matrix(cramer_df):
    # Créer une masque pour la moitié supérieure de la matrice, en excluant la diagonale inférieure
    mask = np.triu(np.ones_like(cramer_df, dtype=bool), k=1)

    plt.figure(figsize=(12, 8))
    sns.heatmap(cramer_df, annot=True, cmap='BuGn', fmt=".2f", mask=mask)
    plt.title("Matrice de corrélation - V de Cramer")
    plt.show()