""" model_meteo module """

import time

from sklearn.model_selection    import GridSearchCV
from sklearn.model_selection    import train_test_split

from sklearn.linear_model       import LogisticRegression
from sklearn.svm                import SVC
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import RandomForestClassifier
from sklearn.ensemble           import GradientBoostingClassifier

from sklearn.pipeline           import Pipeline
from sklearn.preprocessing      import StandardScaler

from imblearn.metrics           import classification_report_imbalanced

from joblib                     import dump, load

import numpy                    as np
import pandas                   as pd


class ModelMeteo:
    """ ModelMeteo class """

    def __init__(self, data):
        """ constructor """
        self.data           = data
        self.x              = None
        self.y              = None
        self.x_train        = None
        self.y_train        = None
        self.x_test         = None
        self.y_test         = None
        self.y_pred         = None
        self.model          = None
        self.params         = { }
        self.results        = { }


    def eval_model(self, model):
        """ evaluate model """
        self.y_pred = self.model.predict(self.x_test)
        self.fill_results(model, self.model)

    def train_model(self, model):
        """ evaluate the model """
        start = time.time()
        match model:
            case "lr":
                self.model  = LogisticRegression(**self.params["lr"])
            case "svm":
                self.model  = SVC(**self.params["svm"])
            case _:
                return "No match"
        self.model.fit(self.x_train, self.y_train)
        end  = time.time()
        self.results[model]["time"] = end - start

    def display_results(self, model):
        """ display results on metrics """
        print(f"Hyperparamètres         : {self.params[model]}\n")
        print(f"Score                   : {self.results[model]["sc"]:.5f}\n")
        print(f"Time training           : {self.results[model]["time"]:.3f}\n")
        print(f"Confusion matrix        :\n\n{self.results[model]["cm"]}\n")
        print(f"Classification report   :\n\n{self.results[model]["cr"]}\n")

    def fill_results(self, name, model):
        """ fill results from the run """
        self.results[name]["sc"]    = model.score(self.x_test, self.y_test)
        self.results[name]["cm"]    = pd.crosstab(self.y_test, self.y_pred, colnames = ["Pred"])
        self.results[name]["cr"]    = classification_report_imbalanced(self.y_test, self.y_pred)

    def hyperparameters_tuning(self, name, model, params, scoring, cv):
        """ find hyperparameters"""
        self.results[name]  = {}
        gridcv              = GridSearchCV(model, params, scoring = scoring, cv = cv, n_jobs = -1)
        gridcv.fit(self.x_train, self.y_train)
        self.params[name]   = gridcv.best_params_

    def init_sets(self, size, rnd):
        """ build train and test datasets """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = size, random_state = rnd)

    def load(self, model):
        """ load model """
        self.model = load(model)

    def persist(self, model):
        """ persist model """
        model_name = model + "saved.joblib"
        dump(self.model, model_name)

    def split_features_target(self, target):
        """ split dataset between features and target """
        self.x = self.data.drop(target, axis = 1)
        self.y = self.data[target]

    def standard_scaler_data(self):
        """ center and scale data """
        sc            = StandardScaler()
        self.x_train  = sc.fit_transform(self.x_train)
        self.x_test   = sc.transform(self.x_test)
