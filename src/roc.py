import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def plot_roc_curve(probabilities, labels, ax, title, color):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    
    probabilities = probabilities.reshape(labels.shape)
    thresholds = np.linspace(0.01, 0.99, 1000)
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
    ax.plot(fpr, tpr, label=title + ' (area = %2.2f)' % mean_auc, lw=2,color=color)
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    ax.set_title('Receiver Operating Characteristic', fontsize=20)
    ax.legend(loc="lower right", fontsize=15)
    ax.plot(thresholds, thresholds, color='k', ls='--', alpha=.5)
        
    return ax, tpr, fpr, thresholds