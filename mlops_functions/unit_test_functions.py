import pandas as pd
from sklearn.model_selection import train_test_split
import unittest
from functions import load_data, filter_summer_season, create_age_classes, create_height_classes, preprocess_data

class TestFunctions(unittest.TestCase):
    def setUp(self):
        # Mettez en place les données que vous utiliserez pour les tests
        self.df = load_data('/Users/Cyrie/OneDrive/Bureau/M2_DS/S1/MLOPS/Rendu_MLOPS/athlete_events.csv')

    def test_load_data(self):
        loaded_data = load_data('/Users/Cyrie/OneDrive/Bureau/M2_DS/S1/MLOPS/Rendu_MLOPS/athlete_events.csv')
        self.assertTrue(loaded_data.equals(self.df))

    def test_filter_summer_season(self):
        summer_data = filter_summer_season(self.df)
        self.assertEqual(summer_data.shape[0], 222552)

    def test_create_age_classes(self):
        age_classes = create_age_classes(self.df)
        expected_classes = ['< 21 ans', '21 - 24 ans', '25-32 ans', '> 32 ans']
        # Ignorez les valeurs 'nan' dans la comparaison
        age_classes = set(age_classes['Classe_age'].dropna())
        expected_classes = set(expected_classes)

        self.assertEqual(age_classes, expected_classes, "Les classes d'âge ne correspondent pas.")

    def test_create_height_classes(self):
        height_classes = create_height_classes(self.df)
        expected_classes = ['< 165 cm', '165 - 172 cm', '173 - 185 cm', '> 185 cm']
        # Ignorez les valeurs 'nan' dans la comparaison
        height_classes = set(height_classes['Classe_height'].dropna())
        expected_classes = set(expected_classes)

        self.assertEqual(height_classes, expected_classes, "Les classes de hauteur ne correspondent pas.")

if __name__ == '__main__':
    unittest.main()
