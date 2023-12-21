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

def load_data(file_path):
    return pd.read_csv(file_path)

def filter_summer_season(df):
    return df[df['Season'] == 'Summer']
