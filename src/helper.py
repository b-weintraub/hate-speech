import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import StandardScaler


def clean_churn_data(data):

    churn = data.copy()
    
    # fix nan in numeric features
    avg_avg_rating_of_driver = np.mean(churn['avg_rating_of_driver'])
    avg_avg_rating_by_driver = np.mean(churn['avg_rating_by_driver'])
    churn['avg_rating_by_driver'] = churn['avg_rating_by_driver'].apply(
        lambda x: avg_avg_rating_by_driver if np.isnan(x) else x)
    churn['avg_rating_of_driver'] = churn['avg_rating_of_driver'].apply(
        lambda x: avg_avg_rating_of_driver if np.isnan(x) else x)
    
    # remove remaining nan
    churn = churn.dropna()
    
    # convert to datetime
    churn['signup_date'] = pd.to_datetime(churn['signup_date'], format='%Y-%m-%d')
    churn['last_trip_date'] = pd.to_datetime(churn['last_trip_date'], format='%Y-%m-%d')
    
    # convert boolean to int
    churn['luxury_car_user'] = churn['luxury_car_user'].astype(int)
    
    #encode phone to boolean (1 for iphone, 0 for android)
    churn['phone'] = churn['phone'].apply(
        lambda x: 1 if x == 'iPhone' else 0)
    
    #one hot encode city
    churn = pd.get_dummies(churn, columns=['city'], prefix='city', prefix_sep=': ', dtype='int')
    
    # add churn column
    churn_date = datetime.strptime('2014-06-01', '%Y-%m-%d')
    churn['churn?'] = (churn['last_trip_date'] < churn_date).astype(int)
    
    # drop date columns
    churn = churn.drop(columns=['signup_date','last_trip_date'])
    
    return churn


def accuracy_scores(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Make calculations
    accuracy = round((tp+tn)/(tp+tn+fp+fn), 3)
    recall = round(tp/(tp+fn), 3)
    precision = round(tp/(tp+fp), 3)
    mse = round(mean_squared_error(y_true, y_pred), 3)
    
    return accuracy, recall, precision, mse


def plot_roc_curve(probabilities, labels, ax, title):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    probabilities = probabilities.reshape(labels.shape)
    thresholds = np.linspace(0.01, 0.99, 100)
    tpr = []
    fpr = []
    
    num_true_pos = np.sum(labels)
    num_false_pos = len(labels) - num_true_pos
    
    for t in thresholds:
        num_correct_pred = np.sum((probabilities >= t) & (labels == 1))
        num_incorrect_pred = np.sum((probabilities >= t) & (labels == 0))
        
        tpr.append(num_correct_pred / num_true_pos)
        fpr.append(num_incorrect_pred / num_false_pos)
        
    mean_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, 'b', label='ROC (area = %2.2f)' % mean_auc, lw=2)
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    ax.set_title('Receiver Operating Characteristic: %s' % title, fontsize=20)
    ax.legend(loc="lower right", fontsize=15)
    ax.plot(thresholds, thresholds, color='k', ls='--', alpha=.5)
        
    return ax#, tpr, fpr, thresholds


def train_test_score(n, model, X_train, X_test, y_train, y_test):
    train_score = np.zeros(n)
    for i, y_pred in enumerate(model.staged_predict(X_train)):
        train_score[i] = model.loss_(y_train, y_pred)

    test_score = np.zeros(n)
    for i, y_pred in enumerate(model.staged_predict(X_test)):
        test_score[i] = model.loss_(y_test, y_pred)
    
    return train_score, test_score
