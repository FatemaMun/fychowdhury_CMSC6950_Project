#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import warnings
from scipy.stats import iqr
warnings.filterwarnings('ignore')
import pandas_bokeh
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
pandas_bokeh.output_notebook()
pd.set_option('plotting.backend', 'pandas_bokeh')
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from bokeh.layouts import row
import pyswarms as ps

class feature_plot:
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def plot_features(self):  
        figure, axs = plt.subplots(5,1,figsize = (17,7))
        color = ['r','g','b','y','m']
        for i in range(self.dataset.shape[1]):
            axs[i].plot(self.dataset.iloc[:,i], label = self.dataset.columns[i], color = color[i])
            axs[i].legend(loc = "lower right")
        figure.tight_layout()
        plt.show()
        
class outlier_removing_with_plots:
    def __init__(self, dataset):
        self.dataset = dataset
    def plot_bar_data(self, dataset):
        figure, axs = plt.subplots(1,dataset.shape[1],figsize = (17,7))
        for i in range(dataset.shape[1]):
            axs[i].boxplot(dataset.iloc[:,i])
            axs[i].set_title(dataset.columns[i])
        plt.show()
    def remove_outliers(self,dataset1):
        for i in range(0,dataset1.shape[1]-1):
            percentile25 = dataset1.iloc[:,i].sort_values().quantile(0.25)
            percentile75 = dataset1.iloc[:,i].sort_values().quantile(0.75)
            iqr2 = iqr(np.array(dataset1.iloc[:,i].sort_values()))
            upper_limit = percentile75 + 1.5 * iqr2
            lower_limit = percentile25 - 1.5 * iqr2
            dataset1 = dataset1[dataset1[dataset1.columns[i]] < upper_limit]
            dataset1 = dataset1[dataset1[dataset1.columns[i]] > lower_limit]
            dataset1.reset_index(drop = True)
        return dataset1
    
class scale_data:
    def __init__(self,dataset):
        self.dataset = dataset
        
    def scaling_dataset(self,scaling_technique):
        if scaling_technique == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            dataset_minmax = scaler.fit_transform(self.dataset)
            return pd.DataFrame(dataset_minmax, columns = self.dataset.columns)
        else:
            from sklearn.preprocessing import StandardScaler
            std_scaler = StandardScaler()
            dataset_std = std_scaler.fit_transform(self.dataset)
            return pd.DataFrame(dataset_std, columns = self.dataset.columns)
        
class model_run:
    def __init__(self, dataset_X, dataset_Y):
        self.dataset_X = dataset_X
        self.dataset_Y = dataset_Y
    def run_model(self, model_name):
        
        if model_name == 'svc':
            model = SVC(kernel = 'rbf')
        if model_name == 'random_forest':
            model = RandomForestClassifier()
    # evaluate the model
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, self.dataset_X, self.dataset_Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
        return np.mean(n_scores)*100
    
def reccursive_feature_elimination(X,y,model_name, number_of_features_to_select):
    
    X = X
    y = y
    
    if model_name == 'svc':
        model = SVC(kernel = 'linear')
        print("Recursive Feature Elimination with Support Vector Machine")

        rfe = RFE(model, number_of_features_to_select)
        fit = rfe.fit(X, y)
        print()
        print("Maximum number of features to be selected: ",number_of_features_to_select)
        print("Name of the selected features: ")
        for j in range(X.shape[1]):
            if fit.ranking_[j] == 1:
                print(X.columns[j])
                
    if model_name == 'random_forest':
        model = RandomForestClassifier()
        print("Recursive Feature Elimination with Random Forest")

        rfe = RFE(model, number_of_features_to_select)
        fit = rfe.fit(X, y)
        print()
        print("Maximum number of features to be selected: ",number_of_features_to_select)
        print("Name of the selected features: ")
        for j in range(X.shape[1]):
            if fit.ranking_[j] == 1:
                print(X.columns[j])

def feature_importance(X,y,model_name):
    if model_name == 'random_forest':
        model = RandomForestClassifier()
        model.fit(X, y)
        print("Feature Importance: ",model.feature_importances_)

        s1 = figure(title = "Using Random Forest",height=350, width=450,x_range = X.columns.to_list())
        s1.vbar(x=X.columns, top=model.feature_importances_, width=0.9,color="#53777a")
        s1.xaxis.axis_label='Feature names'
        s1.yaxis.axis_label='Feature Importance'
        show(s1)
    if model_name == 'svc':
        model = SVC(kernel = 'linear')
        model.fit(X, y)
        print("Feature Importance: ",abs(model.coef_[0]))
        s1 = figure(title = "Using Support Vector Machine",height=350, width=450,x_range = X.columns.to_list())
        s1.vbar(x=X.columns, top=abs(model.coef_[0]), width=0.9,color="#53777a")
        s1.xaxis.axis_label='Feature names'
        s1.yaxis.axis_label='Feature Importance'
        show(s1)
        
def PSO_Feature_selection_with_result(X,y,model_name):
    if model_name == 'random_forest':
        model = RandomForestClassifier()
    if model_name == 'svc':
        model = SVC(kernel = 'rbf')

    def f_per_particle(m, alpha):

        """Computes for the objective function per particle

        Inputs
        ------
        m : numpy.ndarray
            Binary mask that can be obtained from BinaryPSO, will
            be used to mask features.
        alpha: float (default is 0.5)
            Constant weight for trading-off classifier performance
            and number of features

        Returns
        -------
        numpy.ndarray
            Computed objective function
        """
        total_features = 5
        # Get the subset of the features from the binary mask
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X.iloc[:,m==1]
        # Perform classification and store performance in P
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, X_subset, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        # report performance
        j = 1 - np.mean(n_scores)
        #print(j)
        # Compute for the objective function
        #j = (alpha * (1.0 - P)
            #+ (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

        return j

    def f(X, alpha=0.88):
        """Higher-level method to do classification in the
        whole swarm.

        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search

        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        n_particles = X.shape[0]
        j = [f_per_particle(X[i], alpha) for i in range(n_particles)]
        return np.array(j)

    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 15, 'p':2}

    # Call instance of PSO
    dimensions = 5 # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=15, dimensions=dimensions, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(f, iters=20, verbose=2)
    
    # Get the selected features from the final positions
    X_selected_features = X.iloc[:,pos==1]  # subset
    
    # Compute performance
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_selected_features, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    print("Selected Features by PSO: ", np.array(X.columns)[pos == 1])
    return np.mean(n_scores)*100


# In[ ]:




