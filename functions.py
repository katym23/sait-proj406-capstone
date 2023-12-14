import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pyro
import torch
from pyro.infer.mcmc import HMC, MCMC

def visualize_column_distribution(df, list_of_cols):
    """
    Given a dataframe and a list of column names, 
    Plots all the distributions
    """
    
    import plotly.express as px
    fig = []

    for col in list_of_cols:
        fig.append(px.histogram(df, x=col,
                          title = col +  ' distribution'))

    for figure in fig:
        figure.show()
        
def qq_plot(df, list_of_cols):
    
    fig = []
    
    for column in list_of_cols:
        data = df[column]
        sm.qqplot(np.array(data), line='s')
        
    for figure in fig:
        plt.show()
        
def remove_outliers(df, column_name, lower_bound=0.10, upper_bound=0.90):
    """
    Given a df and a column name, filters out outliers
    """
    q1 = df[column_name].quantile(lower_bound)
    q3 = df[column_name].quantile(upper_bound)
    iqr = q3 - q1
    
    lower_threshold = q1 - 1.5 * iqr
    upper_threshold = q3 + 1.5 * iqr
    
    df_filtered = df[(df[column_name] >= lower_threshold) & (df[column_name] <= upper_threshold)]
    
    return df_filtered


def fill_mean(df, column):
    """
    Given a df and a column, fills nulls in the column with the mean
    """
    
    df[column].fillna(df[column].mean(), inplace=True)
    
    return df


def fill_median(df, column):
    """
    Given a df and a column, fills nulls in the column with the median
    """
    
    df[column].fillna(df[column].median(), inplace=True)
    
    return df


def fill_mode(df, column):
    """
    Given a df and a column, fills nulls in the column with the mode
    """
    
    df.loc[:, column] = df[column].fillna(df[column].mode().iloc[0])
    
    return df

def calculate_stats(tn, fp, fn, tp):
    """
    Given the four parts of the matrix:
        tn = true negative
        fp = false positive
        fn = false negative
        tp = true positive,
    returns the four accuracy rates of the individual parts
    e.g. tn rate = tn/total cases
    """
    total = tn + fp + fn + tp
    tn_perc = round((tn/total) * 100, 2)
    fp_perc = round((fp/total) * 100, 2)
    fn_perc = round((fn/total) * 100, 2)
    tp_perc = round((tp/total) * 100, 2)
    
    print('False Negative %: ' + str(fn_perc))
    print('False Positive %: ' + str(fp_perc))
    print('True Negative %: ' + str(tn_perc))
    print('True Positive %: ' + str(tp_perc))
    
    return

def set_threshold(y_pred_proba, threshold=0.24):
    """
    Given a threshold and prediction probabilities,
    transforms the predictions to adhere to the threshold
    """
    return (y_pred_proba[:, 1] > threshold).astype(int)