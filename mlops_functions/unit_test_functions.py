import pandas as pd
from sklearn.model_selection import train_test_split
import unittest
from functions import load_data, filter_summer_season, create_age_classes, create_height_classes, preprocess_data

class TestYourFunctions(unittest.TestCase):
    def setUp(self):
        # Mettez en place les données que vous utiliserez pour les tests
        self.df = load_data('/Users/Cyrie/OneDrive/Bureau/TEST/athlete_events.csv')

    def test_load_data(self):
        loaded_data = load_data('/Users/Cyrie/OneDrive/Bureau/TEST/athlete_events.csv')
        self.assertTrue(loaded_data.equals(self.df))

    def test_filter_summer_season(self):
        summer_data = filter_summer_season(self.df)
        self.assertEqual(summer_data.shape[0], 222552)

    def test_create_age_classes(self):
        age_classes = create_age_classes(self.df)
        self.assertEqual(age_classes['Classe_age'].nunique(), 4)

    def test_create_height_classes(self):
        height_classes = create_height_classes(self.df)
        self.assertEqual(height_classes['Classe_height'].nunique(), 4)


if __name__ == '__main__':
    unittest.main()
