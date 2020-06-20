from flask import Flask, request
from flask_restful import Resource, Api
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder# instantiate labelencoder object
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder# instantiate labelencoder object
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV# Create the parameter grid based on the results of random search
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

app = Flask(__name__)
api = Api(app)


class SimpleLoader:

    def __init__(self, is_training_session=True):
        self._is_training_session = is_training_session

    def load(self, data_path='',
             train_bad_data_path='',
             train_good_data_path='',
             classify_data_path=''):

        for dirname, _, filenames in os.walk(data_path):
            for filename in filenames:
                os.path.join(dirname, filename)

        df = None
        if (self._is_training_session):
            df_bad = pd.read_csv(train_bad_data_path, encoding='ISO-8859-2')
            df_bad.rename(columns={'Unnamed: 0': 'unnamed'}, inplace=True)
            df_bad.drop('unnamed', axis=1, inplace=True)
            df_bad.insert(0, 'label', 0)

            df_good = pd.read_csv(train_good_data_path, encoding='ISO-8859-2')
            df_good.rename(columns={'Unnamed: 0': 'unnamed'}, inplace=True)
            df_good.drop('unnamed', axis=1, inplace=True)
            df_good.insert(0, 'label', 1)
            df = pd.concat([df_good, df_bad], ignore_index=True)
        else:
            df = pd.read_csv(classify_data_path, encoding='ISO-8859-2')
            df.rename(columns={'Unnamed: 0': 'unnamed'}, inplace=True)
            df.drop('unnamed', axis=1, inplace=True)
        return df

class SimplePreProcessor:
    def __init__(self):
        pass

    def preprocess_all(self, df):
        df = self.convert_time(df)
        df = self.convert_numericals(df)
        df = self.encode_categorical(df)
        df = self.normalize_features(df)
        return df

    def encode_categorical(self, df):
        cols = ['ip', 'port']
        le = LabelEncoder()
        df[cols] = df[cols].apply(lambda col: le.fit_transform(col))
        return df

    def normalize_features(self, df):
        cols = ['times', 'di', 'do', 'pi', 'po', 'ip', 'port']
        df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()
        return df

    def convert_time(self, df):
        df[['times']] = df[['times']].apply(self._time_converter, axis=0)
        return df

    def convert_numericals(self, df):
        df[['di']] = df[['di']].apply(self._num_converter, axis=0)
        df[['do']] = df[['do']].apply(self._num_converter, axis=0)
        df[['pi']] = df[['pi']].apply(self._num_converter, axis=0)
        df[['po']] = df[['po']].apply(self._num_converter, axis=0)
        return df

    def _num_converter(self, s):
        s = s.fillna(0).astype(str).str.split(",", expand=False).apply(
            lambda x: [int(y) for y in x]
        )
        s = s.apply(lambda x: np.std(x))
        return s

    def _time_converter(self, s):
        s = s.fillna(0).astype(str).str.split("|", expand=False).apply(
            lambda x: [[int(i) for i in y.split(',')] for y in x]
        )
        s = s.apply(lambda x: np.std(np.concatenate(x)))
        return s

class SimpleModel():
    def __init__(self):
        self._feature_selector_model = None
        self._classifier_model = None

    def train(self, df):
        model, X_train, X_test, X_train, X_test, y_train, y_test = self._selective_train_test_split(df);
        self._feature_selector_model = model
        param_grid = {
            'bootstrap': [True],
            'max_depth': [2, 20, 40],
            'max_features': [1, 3, 7],
            'min_samples_leaf': [2, 4, 8],
            'min_samples_split': [5, 10, 10],
            'n_estimators': [5, 25, 50]
        }
        rf = RandomForestClassifier()
        clf = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2)

        clf.fit(X_train, y_train)
        y_pred = (clf.predict(X_test))
        target_names = ['good', 'bad']
        self._classifier_model = clf

        return (classification_report(y_test, y_pred, target_names=target_names),
                accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred))

    def classify(self, df):
        if self._feature_selector_model is None or self._classifier_model is None:
            raise ('Model appears to be untrained.')

        X = self._feature_selector_model.transform(df[['ip', 'port', 'times', 'di', 'do', 'pi', 'po']])
        y_pred = self._classifier_model.predict(X)
        return y_pred

    def _selective_train_test_split(self, df):
        # Train Test split
        # Feature selection via cheap classifier
        X = df[['ip', 'port', 'times', 'di', 'do', 'pi', 'po']]
        y = df[['label']]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42, shuffle=True, stratify=y)

        # Feature selection via cheap classifier
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X_train, y_train)

        model = SelectFromModel(clf, prefit=True, threshold='0.5*mean')
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)
        return model, X_train, X_test, X_train, X_test, y_train, y_test

class SimplePipeline:
    def __init__(self):
        self._train_loader = SimpleLoader(True)
        self._classify_loader = SimpleLoader(False)
        self._preprocessor = SimplePreProcessor()
        self._simple_model = SimpleModel()

    def fit_csv(self, data_path, train_bad_data_path, train_good_data_path):
        df = self._train_loader.load(data_path=data_path,
                  train_bad_data_path=train_bad_data_path,
                  train_good_data_path = train_good_data_path)
        df = self._preprocessor.preprocess_all(df)
        self._simple_model = SimpleModel()
        return self._simple_model.train(df)


    def predict_csv(self, data_path, classify_data_path):
        df = self._classify_loader.load(classify_data_path=classify_data_path)
        df = self._preprocessor.preprocess_all(df)
        return self._simple_model.classify(df)

    def predict_df(self, df):
        df = self._preprocessor.preprocess_all(df)
        return self._simple_model.classify(df)

class RESTClassifier(Resource):
    def get(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data, index=[0])
        loaded_model = pickle.load(open("model.sav", 'rb'))
        result = loaded_model.predict_df(df)
        return {'prediction': result}

api.add_resource(RESTClassifier, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')