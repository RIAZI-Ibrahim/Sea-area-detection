from ntpath import join
from cv2 import imread
import sklearn.ensemble as ske
import matplotlib.pyplot as plt
from pathlib import Path
from shutil import copyfile
import cv2
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import glob
from PIL import Image
import os
import os.path
import skimage.io as io
import definefun
from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

# Import Support Vector Classifier
from sklearn.svm import SVC
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
svc=SVC(probability=True, kernel='linear')
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
import skimage
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn import datasets, svm, metrics
from sklearn.pipeline import make_pipeline
import os

def compute_labels(rep):
    #labels = []
    dico_labels = []
    for filename in os.listdir(rep)[0:]:
        #print(filename)
        y = 1
        if (int(filename[0]) == 0):
            y =0
        #labels.appendOui
        dico_labels.append(y)
    #return(np.array(labels))
    return dico_labels

X, Y = definefun.contruire_dataset()

    

def getCorrectpredictAlltest(n0,n1):
     y=[]
     for i in range(0,n0):
         y.append(0)
     for i in range(n0,n1):
         y.append(1)   
     return y     

#predict file 
def predictfile(name):
    path = glob.glob(name)
    images = []
    cli=0
    for file in path:
        img = cv2.imread(file)
        images.append(definefun.fd_histogram(img))
        #images.append(definefun.fd_hu_moments(img))
        
    images = np.array(images)    
    x_testp=images
    predictfile=model(x_testp)
    return predictfile


#define our method of Calcul score
def calculscore(y_test,y_predict):
    juste=0
    for i in range(0,len(y_test)):
        if(y_test[i]==y_predict[i]):
            juste=juste+1
    return juste/len(y_test)        



#Notre Modéle pour prédire x_test
def model(x_testp):
    i=0
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i,shuffle=True,test_size=0.6)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(x_train, y_train)
   
    listclassifier.append(clf)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i+1,shuffle=True,test_size=0.6)
    modele_regLog = linear_model.LogisticRegression(solver = 'liblinear', multi_class = 'auto')
#training
    modele_regLog.fit(x_train, y_train)
    
    listclassifier.append(modele_regLog)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i+2,shuffle=True,test_size=0.6)
    classifieur = GaussianNB()
    classifieur.fit(x_train, y_train)
    
    listclassifier.append(classifieur)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i+3,shuffle=True,test_size=0.6)
    classifieur1= SVC(kernel='linear')
    classifieur1.fit(x_train, y_train)
    
    listclassifier.append(classifieur1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i+4,shuffle=True,test_size=0.6)
    clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
    
    clf = clf.fit(x_train, y_train)
    listclassifier.append(clf)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i+5,shuffle=True,test_size=0.6)
   
    rfmodel = ske.RandomForestClassifier(n_estimators = 200,  
                                    bootstrap = True,
                                     verbose = True)

#Run model to assess accuracy
    rf_modelfit = rfmodel.fit(x_train, y_train)
    listclassifier.append(rf_modelfit)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i+6,shuffle=True,test_size=0.6)
    classifieur = GaussianNB()
    classifieur.fit(x_train, y_train)
    listclassifier.append(classifieur)
    y_predict=[]
   
    s=0
    for clf in listclassifier:
        y_predict.append(clf.predict(x_testp))
    predict=[]  
      
    for i in range(0,len(x_testp)):
        predict0=0
        predict1=0
        for k in y_predict:
            if(k[i]==1):
                predict1=predict1+1    
            else:
                predict0=predict0+1
                
        
        if(predict1>predict0):
            predict.append(1)
            

        else:
             predict.append(0)
            
    return predict


#Notre modele qui pour train et tester ou il differe de la fonction prédecedente avec le parametre i 
def train2(x_testp,i):
    listclassifier=[]
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i,shuffle=True,test_size=0.2)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(x_train, y_train)
   
    listclassifier.append(clf)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i+1,shuffle=True,test_size=0.2)
    modele_regLog = linear_model.LogisticRegression(solver = 'liblinear', multi_class = 'auto')
#training
    modele_regLog.fit(x_train, y_train)
    
    listclassifier.append(modele_regLog)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i+2,shuffle=True,test_size=0.2)
    classifieur = GaussianNB()
    classifieur.fit(x_train, y_train)
    
    listclassifier.append(classifieur)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i+3,shuffle=True,test_size=0.2)
    classifieur1= SVC(kernel='linear')
    classifieur1.fit(x_train, y_train)
    
    listclassifier.append(classifieur1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i+4,shuffle=True,test_size=0.2)
    clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
    
    clf = clf.fit(x_train, y_train)
    listclassifier.append(clf)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i+5,shuffle=True,test_size=0.2)
   
    rfmodel = ske.RandomForestClassifier(n_estimators = 200,  
                                    bootstrap = True,
                                     verbose = True)

#Run model to assess accuracy
    rf_modelfit = rfmodel.fit(x_train, y_train)
    listclassifier.append(rf_modelfit)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=i+6,shuffle=True,test_size=0.2)
    classifieur = GaussianNB()
    classifieur.fit(x_train, y_train)
    listclassifier.append(classifieur)
    y_predict=[]
   
    s=0
    for clf in listclassifier:
        y_predict.append(clf.predict(x_testp))
    predict=[]  
      
    for i in range(0,len(x_testp)):
        predict0=0
        predict1=0
        for k in y_predict:
            if(k[i]==1):
                predict1=predict1+1    
            else:
                predict0=predict0+1
                
        
        if(predict1>predict0):
            predict.append(1)
            

        else:
             predict.append(0)
            
    return predict





sum=0   
sum2=0
for i in range(0,10):
    listclassifier=[]
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=20+i,shuffle=True,test_size=0.2)
   # classifieur1= SVC(kernel='linear')
   # classifieur1.fit(x_train, y_train)
    sum2=sum2+calculscore(y_test,train2(x_test,i))
print(sum/10)
print(sum2/10)

print(calculscore(compute_labels("/home/younes/IdeaProjects/iaf/Data/AllTest"),predictfile("/home/younes/IdeaProjects/iaf/Data/AllTest/*")))

"""
"""
"""""
def calculscore(y_test,y_predict):
    juste=0
    for i in range(0,len(y_test)):
        if(y_test[i]==y_predict[i]):
            juste=juste+1
    return juste/len(y_test)        
def train(x_testp,y_testp):
    
    listclassifier=[]
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=0,shuffle=True,test_size=0.2)
  
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(x_train, y_train)
    listclassifier.append(clf)
    modele_regLog = linear_model.LogisticRegression(solver = 'liblinear', multi_class = 'auto')
#training
    modele_regLog.fit(x_train,y_train)
   
    listclassifier.append(modele_regLog)
    
    classifieur = GaussianNB()
    classifieur.fit(x_train, y_train)
    listclassifier.append(classifieur)
    classifieur1= SVC(kernel='linear')
    classifieur1.fit(x_train, y_train)
    listclassifier.append(classifieur1)
    clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
    clf = clf.fit(x_train,y_train)
    listclassifier.append(classifieur1)
    
    
    rfmodel = ske.RandomForestClassifier(n_estimators = 200,  
                                    bootstrap = True,
                                     verbose = True)

#Run model to assess accuracy
    rf_modelfit = rfmodel.fit(x_train, y_train)
    listclassifier.append(rf_modelfit)
    #accuracy = rf_modelfit.score(x_test, y_test)
    classifieur = GaussianNB()
    classifieur.fit(x_train, y_train)
    listclassifier.append(classifieur)
    y_predict=[]
    s=0
    for clf in listclassifier:
        y_predict.append(clf.predict(x_testp))
    predict=[]  
      
    for i in range(0,len(y_test)):
        predict0=0
        predict1=0
        for k in y_predict:
            if(k[i]==1):
                predict1=predict1+1    
            else:
                predict0=predict0+1
                
        
        if(predict1>predict0):
            predict.append(1)

        else:predict.append(0)
        
    return predict
def bagging(x_test,listclassifier):
    y_predict=[]
    s=0
    sum=0
    
    for clf in listclassifier:
        y_predict.append(clf.predict(x_test))
    predict=[]  
      
    for i in range(0,len(x_test)):
        predict0=0
        predict1=0
        for k in y_predict:
            if(k[i]==1):
                predict1=predict1+1    
            else:
                predict0=predict0+1
               
        else:
         if(predict1>predict0):
            predict.append(1)
             
         else:predict.append(0)

        
    return predict

X, Y = definefun.contruire_dataset()

sum=0   
sum2=0
for i in range(0,10):
    listclassifier=[]
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=1,shuffle=True,test_size=0.2)
   # classifieur1= SVC(kernel='linear')
   # classifieur1.fit(x_train, y_train)
    #y_predits1 = classifieur1.predict(x_test)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(x_train, y_train)
    listclassifier.append(clf)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=2,shuffle=True,test_size=0.2)
    modele_regLog = linear_model.LogisticRegression(solver = 'liblinear', multi_class = 'auto')
#training
    modele_regLog.fit(x_train,y_train)
   
    listclassifier.append(modele_regLog)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=3,shuffle=True,test_size=0.2)
    classifieur = GaussianNB()
    classifieur.fit(x_train, y_train)
    listclassifier.append(classifieur)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=4,shuffle=True,test_size=0.2)
    classifieur1= SVC(kernel='linear')
    classifieur1.fit(x_train, y_train)
    listclassifier.append(classifieur1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=5,shuffle=True,test_size=0.2)
    clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
    clf = clf.fit(x_train,y_train)
    listclassifier.append(classifieur1)
    
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=6,shuffle=True,test_size=0.2)
    
    rfmodel = ske.RandomForestClassifier(n_estimators = 200,  
                                    bootstrap = True,
                                     verbose = True)

#Run model to assess accuracy
    rf_modelfit = rfmodel.fit(x_train, y_train)
    listclassifier.append(rf_modelfit)
    #accuracy = rf_modelfit.score(x_test, y_test)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=7,shuffle=True,test_size=0.2)
    classifieur = GaussianNB()
    classifieur.fit(x_train, y_train)
    listclassifier.append(classifieur)
#    sum=sum+bagging(x_test,listclassifier)
   
print(sum/10)

def predictfile(name):
    path = glob.glob(name)
    images = []
    cli=0
    for file in path:
        img = cv2.imread(file)
        #images.append(ImageInParts(img))
        images.append(definefun.fd_histogram(img))

    images = np.array(images)    
    x_testp=images
    predictfile=bagging(x_testp,listclassifier)
    return predictfile

print(calculscore(getCorrectpredictAlltest(40,82),predictfile("/home/younes/IdeaProjects/iaf/Data/AllTest/*")))
#classifieur gaussianNB 
"""

"""
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]
                    
# Apply GridSearchCV to find best parameters for given dataset
# verbose is used to describe the steps taken to find best parameters
cv = GridSearchCV(SVC(), tuned_parameters, refit = True,verbose= 3) 
cv.fit(x_train,y_train)
print("Best parameters to apply are:",cv.best_params_)
# Display model after hyperparameter tuning
svm = cv.best_estimator_
print("Model after tuning is:\n",svm)
y_prediction = svm.predict(x_test)
print("Confusion matrix results:\n",confusion_matrix(y_prediction,y_test))
print("\nClassification report of model:\n",classification_report(y_prediction,y_test))
print("Accuracy score:",100*accuracy_score(y_prediction,y_test))
"""
"""""
classifieur = GaussianNB()
s = 0


for i in range(20):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30, random_state=i)
    classifieur.fit(X_train, Y_train)
    y_predits = classifieur.predict(X_test)
    s = s+classifieur.score(X_test, Y_test)
print (s/20)
   """
#classifieur logistique 
"""""
x_train, x_test, y_train, y_test = train_test_split(X, Y)
#instanciation du modèle
modele_regLog = linear_model.LogisticRegression(solver = 'liblinear', multi_class = 'auto')
#training
modele_regLog.fit(x_train,y_train)
#précision du modèle
precision = modele_regLog.score(x_test,y_test)
print(precision*100)
"""""
#classifieur SVC 

"""
classifier = SVC(kernel='linear', C = 1.0)
classifier.fit(x_train, y_train)
#Prediction sur le Test set
y_pred = classifier.predict(x_test)
print(classifier.score(x_test, y_test))
"""
#classifier tree
"""""

x_train, x_test, y_train, y_test = train_test_split(X, Y)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
"""""
""""
#SGDC CLASSIFIER 
x_train, x_test, y_train, y_test = train_test_split(X, Y)
sgd_clf = SGDClassifier()
sgd_clf.fit(x_train, y_train)
print(sgd_clf.score(x_test,y_test))
"""
"""""
#from internet


print("Les vraies classes :")
print(Y_test)
print("Les classes prédites :")
print(y_predits)

print(classifieur.score(X_test, Y_test))
"""
