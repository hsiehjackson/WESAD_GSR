import numpy as np 
import xgboost as xgb
import os 
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC,LinearSVC


def train(X_train, Y_train, X_val, Y_val, subject, threshold, feature_name,method,feature, classifier):
    #print('traing...')
    train_acc = np.zeros(threshold)
    train_f1 = np.zeros(threshold)
    val_acc = np.zeros(threshold)
    val_f1 = np.zeros(threshold)
    feature_size = np.zeros(threshold)
    predict_all = []
    target_all = []
    xgb1result, train_acc[0], train_f1[0], val_acc[0], val_f1[0], predict, target = modelfit(X_train, Y_train, X_val, Y_val, classifier)
    predict_all.append(predict)
    target_all.append(target)
    for thr in range(threshold):
        if thr==0:
            feature_size[thr] = X_train.shape[1]
            print("feat_size: %.3f train f1: %.3f val f1: %.3f val acc: %.3f " % (feature_size[thr],train_f1[thr], val_f1[thr], val_acc[thr]))
    return train_acc, train_f1, val_acc, val_f1,  feature_size, feature, predict_all, target_all


def modelfit(X_train, Y_train, X_val, Y_val, classifier='SVM'):
    
    scoring = 'neg_log_loss'

    if classifier == 'SVM':
        model = SVC(probability=True)
        param_grid = {'kernel':['linear', 'rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [1000 ,100,10,1,1e-1,1e-2,1e-3, 1e-4, 'auto']}
        gsearch = GridSearchCV(estimator=model,param_grid=param_grid, scoring=scoring, n_jobs=-1,iid=False,cv=10,verbose=0)
        gsearch.fit(X_train,Y_train)
        model = gsearch.best_estimator_

    elif classifier == 'kNN':
        model = KNeighborsClassifier()
        param_grid = {'n_neighbors':list(range(1, 30)),'weights': ['uniform', 'distance'] }
        gsearch = GridSearchCV(estimator=model,param_grid=param_grid, scoring=scoring, n_jobs=-1,iid=False,cv=10,verbose=0)
        gsearch.fit(X_train,Y_train)
        model = gsearch.best_estimator_

    elif classifier == 'DT':
        model = DecisionTreeClassifier()
        param_grid = {'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
        gsearch = GridSearchCV(estimator=model,param_grid=param_grid, scoring=scoring, n_jobs=-1,iid=False,cv=10,verbose=0)
        gsearch.fit(X_train,Y_train)
        model = gsearch.best_estimator_

    elif classifier == 'RF':
        model = RandomForestClassifier()
        param_grid = {  'bootstrap': [True],
                        'max_depth': [80, 90, 100, 110],
                        'max_features': [2, 3],
                        'min_samples_leaf': [3, 4, 5],
                        'min_samples_split': [8, 10, 12],
                        'n_estimators': [100, 200, 300, 1000]}
        gsearch = GridSearchCV(estimator=model,param_grid=param_grid, scoring=scoring, n_jobs=-1,iid=False,cv=10,verbose=0)
        gsearch.fit(X_train,Y_train)
        model = gsearch.best_estimator_

    elif classifier == 'AD':
        model = DecisionTreeClassifier()
        param_grid = {'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
        gsearch = GridSearchCV(estimator=model,param_grid=param_grid, scoring=scoring, n_jobs=-1,iid=False,cv=10,verbose=0)
        gsearch.fit(X_train,Y_train)
        model = gsearch.best_estimator_
        model = AdaBoostClassifier(base_estimator=model)
        param_grid = {  "base_estimator__criterion": ["gini", "entropy"],
                        "base_estimator__splitter":   ["best", "random"],
                        "n_estimators": [100, 200, 300, 1000]}
        gsearch = GridSearchCV(estimator=model,param_grid=param_grid, scoring=scoring, n_jobs=-1,iid=False,cv=10,verbose=0)
        gsearch.fit(X_train,Y_train)
        model = gsearch.best_estimator_
        model.fit(X_train, Y_train)

    Ytrain_pred_prob  = model.predict_proba(X_train)
    Yval_pred_prob = model.predict_proba(X_val)
    Ytrain_pred = np.argmax(Ytrain_pred_prob,axis=1)
    Yval_pred = np.argmax(Yval_pred_prob,axis=1)
    train_acc = accuracy_score(Y_train,Ytrain_pred) 
    train_f1 = f1_score(Y_train,Ytrain_pred,average='macro')
    train_loss = log_loss(Y_train,Ytrain_pred_prob)
    val_acc = accuracy_score(Y_val,Yval_pred)
    val_f1 = f1_score(Y_val,Yval_pred,average='macro')
    val_loss = log_loss(Y_val, Yval_pred_prob)

    return model, train_acc, train_f1, val_acc, val_f1, Yval_pred, Y_val