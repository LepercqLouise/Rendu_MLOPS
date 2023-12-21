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



#Fonction pour ACM 
def analyze_acm(df):
    # Remplacer les valeurs manquantes de la variable 'Medal' par 'Pas_de_médaille'
    df['Medal'].fillna('Pas_de_médaille', inplace=True)

    # Utilisation du package fanalysis - MCA
    acm = MCA()
    acm.fit(df.values)

    # Afficher les valeurs propres
    eigenvalues = acm.eig_

    # Calculer le pourcentage de chaque valeur propre
    total_variance = sum(eigenvalues[0])
    percentage_var = [(value / total_variance) * 100 for value in eigenvalues[0]]

    # Tracer le diagramme en barres des valeurs propres en pourcentage
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, len(percentage_var) + 1), percentage_var, color=palette, edgecolor='black')
    plt.xlabel('Composante Principale')
    plt.ylabel('Pourcentage de Variance Expliquée')
    plt.title('Diagramme en Barres des Valeurs Propres en Pourcentage - ACM')
    plt.xticks(range(1, len(percentage_var) + 1))

    # Ajouter les étiquettes de valeur uniquement pour les deux premières barres
    for i, (bar, value) in enumerate(zip(bars, percentage_var)):
        if i < 2:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f'{value:.2f}%',
                     ha='center', va='bottom', color='black', fontsize=8)

    plt.show()

    # Information fournie par l'ACM
    info_col = acm.col_topandas()
    print(info_col.columns)

    # Coordonnées des modalités pour l'axe 1 et 2
    coord_col = info_col[['col_coord_dim1', 'col_coord_dim2']]
    print(coord_col)

    # Contributions des modalités pour l'axe 1 et 2
    contrib_col = pd.DataFrame(info_col[['col_contrib_dim1', 'col_contrib_dim2']])
    print(contrib_col)

    # ACM - Projection des colonnes
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis([-2.2, +2.2, -1.2, +1.5])
    ax.plot([-2.2, +2.2], [0, 0], color="silver", linestyle="--")
    ax.plot([0, 0], [-2.2, +2.2], color='silver', linestyle="--")
    ax.set_xlabel('Dim.1')
    ax.set_ylabel('Dim.2')
    plt.title("Modalité")

    for x, y, lbl in zip(coord_col.iloc[:, 0], coord_col.iloc[:, 1], coord_col.index):
        ax.text(x, y, lbl, horizontalalignment='center', verticalalignment='center', fontsize=7)

    plt.show()

    # ACM - Projection en couleur
    acm.mapping_col(num_x_axis=1, num_y_axis=2)


#Fonction pour modelisation
def preprocess_and_split_data(base_model):
    # Affichage des valeurs uniques avant la modification
    print("Valeurs uniques avant la modification :", base_model['Medal'].unique())

    # Regroupement des 3 médailles en une modalité médaille
    base_model['Medal'].replace({
        'Gold': 'medaille',
        'Silver': 'medaille',
        'Bronze': 'medaille',
    }, inplace=True)

    # Affichage des valeurs uniques après la modification
    print("Valeurs uniques après la modification :", base_model['Medal'].unique())

    # y variable expliquée et X les variables explicatives
    y = base_model.Medal
    X = base_model.drop('Medal', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Cette étape permet de ne pas créer des dummy pour réaliser les modèles
    cat_columns = ['Sex', 'NOC', 'Sport', 'Classe_age', 'Classe_height', 'Classe_weight']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), cat_columns)
        ],
        remainder='drop'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    return X_train_encoded, X_test_encoded, y_train, y_test


#Fonction regression logistique 
def train_and_evaluate_logistic_regression(X_train_encoded, X_test_encoded, y_train, y_test):
    # Initialisation du modèle de régression logistique
    LG = LogisticRegression(random_state=0, max_iter=10000)

    # Entraînement du modèle
    LG.fit(X_train_encoded, y_train)

    # Score sur les données d'entraînement
    training_score = LG.score(X_train_encoded, y_train)
    print("Training score:", training_score)

    # Score sur les données de test
    test_score = LG.score(X_test_encoded, y_test)
    print("Test score:", test_score)

    # Prédiction sur les données d'entraînement et de test
    y_train_pred = LG.predict(X_train_encoded)
    y_test_pred = LG.predict(X_test_encoded)

    # Matrice de confusion pour les données d'entraînement
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    print("\nMatrice de confusion (Training Data):")
    print(conf_matrix_train)

    # Matrice de confusion pour les données de test
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    print("\nMatrice de confusion (Test Data):")
    print(conf_matrix_test)

    # Calcul des indicateurs de performance du modèle
    TP = conf_matrix_test[0, 0]
    FP = conf_matrix_test[1, 0]
    FN = conf_matrix_test[0, 1]
    TN = conf_matrix_test[1, 1]

    sensibilite = round((TP / (TP + FN)), 2) * 100
    specificite = round((TN / (TN + FP)), 2) * 100
    taux_erreur_alpha = round((FN / (TP + FN)), 2) * 100
    taux_erreur_beta = (FP / (FP + TN)) * 100
    taux_erreur_moyen = round(((FP + FN) / (FP + FN + TP + TN)), 2) * 100

    print("\nIndicateurs de performance du modèle :")
    print("Sensibilité:", sensibilite)
    print("Spécificité:", specificite)
    print("Taux d'erreur alpha:", taux_erreur_alpha)
    print("Taux d'erreur beta:", taux_erreur_beta)
    print("Taux d'erreur moyen:", taux_erreur_moyen)

    return LG  # Retourne le modèle entraîné

#Focntion Random Forest

def train_and_evaluate_random_forest(X_train_encoded, X_test_encoded, y_train, y_test):
    # Créer une instance du modèle Random Forest
    RF = RandomForestClassifier(random_state=0, n_estimators=100)

    # Entraîner le modèle sur les données d'entraînement encodées
    RF.fit(X_train_encoded, y_train)

    # Score sur les données d'entraînement
    training_score_RF = RF.score(X_train_encoded, y_train)
    print("Training score:", training_score_RF)

    # Score sur les données de test
    test_score_RF = RF.score(X_test_encoded, y_test)
    print("Test score:", test_score_RF)

    # Prédiction sur les données d'entraînement et de test
    y_train_pred_RF = RF.predict(X_train_encoded)
    y_test_pred_RF = RF.predict(X_test_encoded)

    # Matrice de confusion pour les données d'entraînement
    conf_matrix_train_RF = confusion_matrix(y_train, y_train_pred_RF)
    print("\nMatrice de confusion (Training Data):")
    print(conf_matrix_train_RF)

    # Matrice de confusion pour les données de test
    conf_matrix_test_RF = confusion_matrix(y_test, y_test_pred_RF)
    print("\nMatrice de confusion (Test Data):")
    print(conf_matrix_test_RF)

    # Calcul des indicateurs de performance du modèle
    TP_RF = conf_matrix_test_RF[0, 0]
    FP_RF = conf_matrix_test_RF[1, 0]
    FN_RF = conf_matrix_test_RF[0, 1]
    TN_RF = conf_matrix_test_RF[1, 1]

    sensibilite_RF = round((TP_RF / (TP_RF + FN_RF)), 2) * 100
    specificite_RF = round((TN_RF / (TN_RF + FP_RF)), 2) * 100
    taux_erreur_alpha_RF = round((FN_RF / (TP_RF + FN_RF)), 2) * 100
    taux_erreur_beta_RF = (FP_RF / (FP_RF + TN_RF)) * 100
    taux_erreur_moyen_RF = round(((FP_RF + FN_RF) / (FP_RF + FN_RF + TP_RF + TN_RF)), 2) * 100

    print("\nIndicateurs de performance du modèle :")
    print("Sensibilité:", sensibilite_RF)
    print("Spécificité:", specificite_RF)
    print("Taux d'erreur alpha:", taux_erreur_alpha_RF)
    print("Taux d'erreur beta:", taux_erreur_beta_RF)
    print("Taux d'erreur moyen:", taux_erreur_moyen_RF)

    return RF  


#Fonction knn

def choose_k_and_evaluate_knn(X_train_encoded, X_test_encoded, y_train, y_test, max_k=50):
    # Choix du k optimal
    error_rate = []
    k_values = list(filter(lambda x: x % 2 == 1, range(1, max_k + 1, 2)))

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_encoded, y_train)
        pred_i = knn.predict(X_test_encoded)
        error_rate.append(np.mean(pred_i != y_test))

    best_k_index = error_rate.index(np.min(error_rate))
    best_k = k_values[best_k_index]
    print("Meilleur k:", best_k)

    # Figure qui montre le k optimal
    plt.figure(figsize=(10, 10))
    plt.plot(k_values, error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()

    # Créer une instance du modèle k-NN avec le meilleur k
    knn = KNeighborsClassifier(n_neighbors=best_k)

    # Entraîner le modèle sur les données d'entraînement encodées
    knn.fit(X_train_encoded, y_train)

    # Score sur les données d'entraînement
    training_score_knn = knn.score(X_train_encoded, y_train)
    print("\nTraining score:", training_score_knn)

    # Score sur les données de test
    test_score_knn = knn.score(X_test_encoded, y_test)
    print("Test score:", test_score_knn)

    # Prédiction sur les données d'entraînement et de test
    y_train_pred_knn = knn.predict(X_train_encoded)
    y_test_pred_knn = knn.predict(X_test_encoded)

    # Matrice de confusion pour les données d'entraînement
    conf_matrix_train_knn = confusion_matrix(y_train, y_train_pred_knn)
    print("\nMatrice de confusion (Training Data):")
    print(conf_matrix_train_knn)

    # Matrice de confusion pour les données de test
    conf_matrix_test_knn = confusion_matrix(y_test, y_test_pred_knn)
    print("\nMatrice de confusion (Test Data):")
    print(conf_matrix_test_knn)

    # Calcul des indicateurs de performance du modèle
    TP_knn = conf_matrix_test_knn[0, 0]
    FP_knn = conf_matrix_test_knn[1, 0]
    FN_knn = conf_matrix_test_knn[0, 1]
    TN_knn = conf_matrix_test_knn[1, 1]

    sensibilite_knn = round((TP_knn / (TP_knn + FN_knn)), 2) * 100
    specificite_knn = round((TN_knn / (TN_knn + FP_knn)), 2) * 100
    taux_erreur_alpha_knn = round((FN_knn / (TP_knn + FN_knn)), 2) * 100
    taux_erreur_beta_knn = (FP_knn / (FP_knn + TN_knn)) * 100
    taux_erreur_moyen_knn = round(((FP_knn + FN_knn) / (FP_knn + FN_knn + TP_knn + TN_knn)), 2) * 100

    print("\nIndicateurs de performance du modèle :")
    print("Sensibilité:", sensibilite_knn)
    print("Spécificité:", specificite_knn)
    print("Taux d'erreur alpha:", taux_erreur_alpha_knn)
    print("Taux d'erreur beta:", taux_erreur_beta_knn)
    print("Taux d'erreur moyen:", taux_erreur_moyen_knn)

    return knn 