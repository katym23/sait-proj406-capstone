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

    Parameters
    df (pandas dataframe): the dataframe that contains
    the columns to be visualized
    list_of_cols (list): a list containing the columns
    to be visualized. Each column name should be a string.

    Returns
    plotly figures of the distributions of values
    within each column in list_of_cols
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


def fill_mean(df, column):
    """
    Given a df and a column, fills nulls in the column with the mean
    
    Parameters
    df (pandas dataframe): the dataframe that contains
    the column to be filled
    column (str): the column to be filled

    Returns
    df (pandas dataframe) a dataframe with the nulls in 
    column filled
    """
    
    df[column].fillna(df[column].mean(), inplace=True)
    
    return df


def fill_median(df, column):
    """
    Given a df and a column, fills nulls in the column with the median

    Parameters
    df (pandas dataframe): the dataframe that contains
    the column to be filled
    column (str): the column to be filled

    Returns
    df (pandas dataframe) a dataframe with the nulls in 
    column filled
    """
    
    df[column].fillna(df[column].median(), inplace=True)
    
    return df


def fill_mode(df, column):
    """
    Given a df and a column, fills nulls in the column with the mode

    Parameters
    df (pandas dataframe): the dataframe that contains
    the column to be filled
    column (str): the column to be filled

    Returns
    df (pandas dataframe) a dataframe with the nulls in 
    column filled
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

    Parameters
    tn (int): num of true negatives
    fp (int): num of false positives
    fn (int): num of false negatives
    tp (int): num of true positives

    Returns
    printed statements with the percentage values
    for each input
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
    transforms the predictions to labels that
    adhere to the given threshold

    Parameters
    y_pred_proba (array): predicted y probabilities
    threshold (float): the desired threshold

    Returns
    an array of predicted values that adhere to the 
    threshold
    """
    return (y_pred_proba[:, 1] > threshold).astype(int)
