import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv, DataFrame
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_df(f):
    df = read_csv(
        './15_LetterRecognition/letter-recognition.data',
        header=None,
        delimiter=','
    )

    def inner():
        f(df)

    return inner

@load_df
def process(df: DataFrame):
    x = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    accuracy_ = dict()
    for i in range(1, 15):
        classifier = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        ac = accuracy_score(y_test, y_pred)
        accuracy_[i] = ac

    for k, v in accuracy_.items():
        print(f"{k} neighbors -> {v}")
    idx = max(accuracy_, key=accuracy_.get)
    print(f"Best: {idx} neighbors ({accuracy_[idx]})")


def main():
    process()


if __name__ == '__main__':
    main()
