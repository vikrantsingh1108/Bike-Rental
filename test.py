import unittest
from sklearn.ensemble import RandomForestRegressor
from Models import random_forestRegressorClassifier
from main import dataSet_reader
import sys
import io
from os import path


class Test_Methods(unittest.TestCase):
    
    def setUp(self):
        ''' Test runner setup.
        ''' 
        self.datareader = dataSet_reader()
        self.mdl = random_forestRegressorClassifier()
        self.mdl.model_file = 'model.pth'

    def test_Variables(self):
        ''' Test whether feature and target variables are defined in the model.
        '''
        self.assertGreater(len(self.datareader.feature_data), 0)
        self.assertGreater(len(self.datareader.target), 0)

    def test_Model_Saved(self):
        ''' Test the save method for a blank model.
        '''
        self.mdl.model = RandomForestRegressor()
        self.assertTrue(self.mdl.save_model())

    def test_NonExistingModel(self):
        ''' Test the save method for a non existing model.
        '''
        self.mdl.classifier = None
        self.assertFalse(self.mdl.save_model())

    def test_Exisitng_Model_loading(self):
        ''' Test the load model for a previous saved model.
        '''
        self.mdl.save_model()
        self.assertTrue(self.mdl.load_Model())

    def test_nonExisitng_Model_loading(self):
        ''' Test the load method for a non-existing model path.
        '''
        invalid_file = 'not available'
        self.assertFalse(self.mdl.load_Model(invalid_file))

if __name__ == '__main__':
    unittest.main()