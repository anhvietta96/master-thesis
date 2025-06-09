#!/usr/bin/env python3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
import argparse
import time
import pandas as pd


def parse_command_line(argv):
    p = argparse.ArgumentParser(description='Split a hdf5 dataset')
    p.add_argument('-d', '--debug', action='store_true', default=False,
                   help='show debug output')
    p.add_argument("-i", "--input", nargs=1, type=str,
                   required=True,  help="Specify the input output.")
    return p.parse_args(argv)


def random_forest(inputfile):
    df = pd.read_csv(inputfile, sep='\t')
    X = df[['Alignment length', 'Identity']]
    y = df['Matched']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    parameters = {'n_estimators': [10, 15], 'criterion': [
        'gini', 'entropy', 'log_loss'], 'max_depth': [10, 15]}
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, parameters)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test,y_pred)}')
    print(f'Confusion matrix: {confusion_matrix(y_test,y_pred)}')
    best_estimator = clf.best_estimator_
    print(best_estimator.feature_importances_)
    print(best_estimator.get_params())


if __name__ == '__main__':
    args = parse_command_line(sys.argv[1:])
    t = time.time()
    random_forest(args.input[0])
    print(f'Time to compare data: {time.time()-t}')
