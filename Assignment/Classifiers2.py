# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 23:32:39 2016

@author: AAdhikari
@description: A sample code to implement SVM classifier by tuning its parameter 
using grid search and cross validation on the traing samples. The scikit-learn 
module is used. 
"""
import numpy as np
#from sklearn.datasets import make_moons
import csv
import warnings
from sklearn import svm
from sklearn import grid_search  
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
warnings.simplefilter("ignore",category=DeprecationWarning)
if __name__ == '__main__':
    print 'started\n'
    scaler = MinMaxScaler();
    fread = open('reut_train_features.csv','rb')
    data = list(csv.reader(fread,delimiter = ','))
    fread.close()
    print 'Train read complete\n'
    y_train = [item[1] for item in data]
    X_train = data
    for row in X_train:
        del row[1]  
    X_tscale = np.array(X_train, dtype = float) 
    y_train = np.array(y_train)   
    X_train = scaler.fit_transform(X_tscale)
    #y_train= preprocessing.scale(y_tscale)
    print 'Train split complete\n'
    fread = open('reut_test_features.csv','rb')
    data = list(csv.reader(fread,delimiter = ','))
    fread.close()
    print 'Test read complete\n'
    y_test = [item[1] for item in data]
    X_test = data
    for row in X_test:
        del row[1]  
    X_testscale = np.array(X_test, dtype = float)        
    y_test = np.array(y_test)  
    X_test = scaler.fit_transform(X_testscale)
    #y_test=preprocessing.scale(y_testscale)
 
    print 'Test split complete\n'
        #Naive Bayes
    print 'Running NB'
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    print 'Classification of the train samples'
    predicted_class_label = clf.predict(X_train)     
    predicted_class_label = list(predicted_class_label)
    
    print 'Evaluation using Precision, Recall and F-measure'  
    pr=precision_score(y_train, predicted_class_label, average='micro')
    print '\n Precision:'+str(pr)
    re=recall_score(y_train, predicted_class_label, average='micro')
    print '\n Recall:'+str(re)
    fm=f1_score(y_train, predicted_class_label, average='micro') 
    print '\n F-measure:'+str(fm)
    
    print 'Classification of the test samples'
    predicted_class_label = clf.predict(X_test)     
    predicted_class_label = list(predicted_class_label)
    
    print 'Evaluation using Precision, Recall and F-measure'  
    pr=precision_score(y_test, predicted_class_label, average='micro')
    print '\n Precision:'+str(pr)
    re=recall_score(y_test, predicted_class_label, average='micro')
    print '\n Recall:'+str(re)
    fm=f1_score(y_test, predicted_class_label, average='micro') 
    print '\n F-measure:'+str(fm)
'''
    # SVM Classifier 
    print 'running svm'
    svr = svm.SVC(class_weight='balanced')
    param_grid =[{'kernel':['linear'],'C':[int(i) for i in range(1,100)]},{'kernel':['poly','rbf'],'C':[1,10,100]},]  # Sets of parameters
    grid = grid_search.GridSearchCV(svr,param_grid,n_jobs=4, cv=5)          
    grid.fit(X_train,y_train)    
    clf= grid.best_estimator_                   # Best grid
    print '\n The best grid is as follows: \n'
    print grid.best_estimator_ 

    print 'Classification of the train samples'
    predicted_class_label = clf.predict(X_train)     
    predicted_class_label = list(predicted_class_label)

    print 'Evaluation using Precision, Recall and F-measure'  
    pr=precision_score(y_train, predicted_class_label, average='micro')
    print '\n Precision:'+str(pr)
    re=recall_score(y_train, predicted_class_label, average='micro')
    print '\n Recall:'+str(re)
    fm=f1_score(y_train, predicted_class_label, average='micro') 
    print '\n F-measure:'+str(fm)
 
    print 'Classification of the test samples'
    predicted_class_label = clf.predict(X_test)     
    predicted_class_label = list(predicted_class_label)

    print 'Evaluation using Precision, Recall and F-measure'  
    pr=precision_score(y_test, predicted_class_label, average='micro')
    print '\n Precision:'+str(pr)
    re=recall_score(y_test, predicted_class_label, average='micro')
    print '\n Recall:'+str(re)
    fm=f1_score(y_test, predicted_class_label, average='micro') 
    print '\n F-measure:'+str(fm)

    print 'running LR'
    #Logistic Regression
    logReg = LogisticRegression(class_weight = 'balanced',penalty = 'l2')
    param_grid = {'C': [1e6, 1e5, 1e4 , 0.001, 0.01, 0.1, 1] }
    grid = grid_search.GridSearchCV(logReg,param_grid,n_jobs=4,cv=10)          
    grid.fit(X_train,y_train)    
    clf= grid.best_estimator_                   # Best grid
    print '\n The best grid is as follows: \n'
    print grid.best_estimator_ 
    
    print 'Classification of the train samples'
    predicted_class_label = clf.predict(X_train)     
    predicted_class_label = list(predicted_class_label)
    
    print 'Evaluation using Precision, Recall and F-measure'  
    pr=precision_score(y_train, predicted_class_label, average='micro')
    print '\n Precision:'+str(pr)
    re=recall_score(y_train, predicted_class_label, average='micro')
    print '\n Recall:'+str(re)
    fm=f1_score(y_train, predicted_class_label, average='micro') 
    print '\n F-measure:'+str(fm)
    
    print 'Classification of the test samples'
    predicted_class_label = clf.predict(X_test)     
    predicted_class_label = list(predicted_class_label)
    
    print 'Evaluation using Precision, Recall and F-measure'  
    pr=precision_score(y_test, predicted_class_label, average='micro')
    print '\n Precision:'+str(pr)
    re=recall_score(y_test, predicted_class_label, average='micro')
    print '\n Recall:'+str(re)
    fm=f1_score(y_test, predicted_class_label, average='micro') 
    print '\n F-measure:'+str(fm)
    
    #Multi-Layer Perceptron
    print 'running MLP'
    clf = MLPClassifier(solver='adam',alpha = 1e-05,random_state=1)
    hid = [(20 for i in range(30)),(i for i in range(30))]                    
    param_grid = {'activation':['identity','logistic', 'tanh', 'relu']}
    grid = grid_search.GridSearchCV(clf, param_grid,n_jobs=8,cv=10)
    grid.fit(X_train,y_train)    
    clf= grid.best_estimator_                   # Best grid
    print '\n The best grid is as follows: \n'
    print grid.best_estimator_ 
    
    print 'Classification of the train samples'
    predicted_class_label = clf.predict(X_train)     
    predicted_class_label = list(predicted_class_label)
    
    print 'Evaluation using Precision, Recall and F-measure'  
    pr=precision_score(y_train, predicted_class_label, average='micro')
    print '\n Precision:'+str(pr)
    re=recall_score(y_train, predicted_class_label, average='micro')
    print '\n Recall:'+str(re)
    fm=f1_score(y_train, predicted_class_label, average='micro') 
    print '\n F-measure:'+str(fm)
    
    print 'Classification of the test samples'
    predicted_class_label = clf.predict(X_test)     
    predicted_class_label = list(predicted_class_label)
    
    print 'Evaluation using Precision, Recall and F-measure'  
    pr=precision_score(y_test, predicted_class_label, average='micro')
    print '\n Precision:'+str(pr)
    re=recall_score(y_test, predicted_class_label, average='micro')
    print '\n Recall:'+str(re)
    fm=f1_score(y_test, predicted_class_label, average='micro') 
    print '\n F-measure:'+str(fm)
'''
