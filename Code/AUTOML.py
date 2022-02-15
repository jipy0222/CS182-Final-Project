import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor


def AUTOML(data):
    train_data, test_data = train_test_split(data, test_size=0.1)
    label = 'rating'
    average = train_data[label].mean()
    y_true = test_data[label]
    predictor = TabularPredictor(label=label).fit(
        train_data=train_data, excluded_model_types=['NN'])
    predictor.fit_summary(show_plot=True)
    y_pred = predictor.predict(test_data.drop(columns=[label]))
    r = y_pred.shape[0]
    c = 1

    y_pred2 = np.ones((r, c))*average
    y_pred3 = np.random.rand(r, c)*5
    rmse1 = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse2 = np.sqrt(mean_squared_error(y_true, y_pred2))
    rmse3 = np.sqrt(mean_squared_error(y_true, y_pred3))
    mae1 = mean_absolute_error(y_true, y_pred)
    mae2 = mean_absolute_error(y_true, y_pred2)
    return [rmse1, rmse2, rmse3, mae1, mae2]


if __name__ == '__main__':
    data1 = TabularDataset('data/ratings_full_1.csv')
    print("\n", AUTOML(data1))
    data2 = TabularDataset('data/ratings_full_2.csv')
    print("\n", AUTOML(data2))

#[0.8950206541477348, 1.0083311185434676, 1.9118292936738492, 0.7023999283188268, 0.8025929927457941]
#[0.6728123640530519, 0.8117154223705839, 2.1515192990271905, 0.5258329931155655, 0.6465984482741315]
