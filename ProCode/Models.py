import abc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import os


class AbstractModelClass(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.classifier =0
        pass

    @staticmethod
    def populate_classifier_models():
        classifier_models={}
        classifier_models["random_forestClassifier"] = randForestClassifier
        classifier_models["linear_regressionModel"] = linear_regressionModel
        classifier_models["random_forestRegressorClassifier"] =random_forestRegressorClassifier
        # add more classifier models  
        return classifier_models


class randForestClassifier(AbstractModelClass):

    def __init__(self,n_estimators):
        self.n_estimators=n_estimators
        self.create_model()

    def create_model(self):
        self.classifier =  SelectFromModel(RandomForestClassifier(self.n_estimators))

    def fit(self,data,labels):
        self.classifier.fit(data,labels)

    def classifier_prediction(self,x_test):
        return self.classifier.predict(x_test)

    def get_support(self):
        return self.classifier.get_support()

    def get_SelectedFeatures(self, feature_data ):
        return feature_data.columns[self.get_support()]

    def get_FeatureImportance(self):
        feature_ranked = self.classifier.estimator_.feature_importances_
        ranked_feature_indices = np.argsort(feature_ranked)[::-1]
        return ranked_feature_indices , feature_ranked



class linear_regressionModel(AbstractModelClass):

    def __init__(self):
        self.create_model()

    def create_model(self):
        self.classifier =  LinearRegression()

    def fit(self,data,labels):
        self.classifier.fit(data,labels)

    def classifier_prediction(self,x_test):
        return self.classifier.predict(x_test)


class random_forestRegressorClassifier(AbstractModelClass):

    def __init__(self,n_estimators=300):
        self.create_model(n_estimators)

    def create_model(self,n_estimators):
        self.classifier =  RandomForestRegressor(n_estimators=n_estimators , max_features = 'auto')

    def fit(self,data,labels):
        self.classifier.fit(data,labels)

    def classifier_prediction(self,x_test):
        return self.classifier.predict(x_test)

    def save_model(self, model_file='model.pth'):
        success = False
        if self.classifier is not None:
            pickle.dump(self.classifier, open(model_file, 'wb'))
            success = True
        return success

    def load_Model(self, model_file='model.pth'):
        success = False
        if os.path.exists(model_file):
            self.classifier = pickle.load(open(model_file, 'rb'))
            success = True
        return success