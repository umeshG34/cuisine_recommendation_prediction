#!/usr/bin/python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
import warnings
import argparse
import nltk
import re
import json
import os
#import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric

def get():
    with open('yummly.json',encoding = 'utf-8') as json_data:
        print('-->Reading JSON file.')
        list_rec = json.load(json_data)
        #print("-->First Recipe:",list_rec[0])
        print("-->Number of Recipes:",len(list_rec))
    return list_rec

def eda(list_rec):
    rec_ings = []
    u_ingd = []
    X,y = [],[]
    ids = []
    #looking at number ingredients etc.
    for i in list_rec:
        ids.append(i['id'])
        rec_ings.append(i['ingredients'])
        u_ingd.extend(i['ingredients'])
        y.append(i['cuisine'])
    tot_ings = len(u_ingd)
    u_ingd = list(set(u_ingd))
    print("-->Number of Unique Ingredients:",len(u_ingd))#,"\n-->Average number of ingredients per recipe:",tot_ings/len(list_rec))
    return y, ids, rec_ings
    
def train_mod(list_rec):
    y, ids, rec_ings = eda(list_rec)
    print('-->Obtained Train and Target')
    #Combining all ingredients in weach reciepe for input to tf-idf
    temp_ings = []
    for ind, ings in enumerate(rec_ings):
        all_ing ='þ' + 'þþ'.join(ings) + 'þ'
        temp_ings.append(all_ing)
        #print(all_ing)
    #print(temp_ings)
    #print("Cuisines",set(y)) 
    #Tf-idf for better features.
    with warnings.catch_warnings(): 
        warnings.filterwarnings(action = "ignore",category = FutureWarning)
        #vectorize = TfidfVectorizer(token_pattern=r'þ(.*?)þ',min_df =0.0001)
        vectorize = CountVectorizer(token_pattern = r'þ(.*?)þ',min_df = 0.0001)
        model = vectorize.fit_transform(temp_ings)
    #print("Features:",vectorize.get_feature_names())
    X = model.toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05,random_state = 1312)
    print("Dimensions of Input:",X.shape)
    print("-->Training the model for predicting Cuisine.")
    clf = MultinomialNB().fit(X_train,y_train)
    #clf = BernoulliNB().fit(X_train,y_train)
    #clf = MLPClassifier(hidden_layer_sizes = (200,100,50),verbose = True,max_iter = 50).fit(X_train,y_train)
    print("Train Accuracy:",clf.score(X_train,y_train),"Test Accuracy:",clf.score(X_test,y_test))
    return clf, vectorize, ids, X

def predict_cus(list_rec):
    filename = "data/MLP.sav"
    clf , vectorize, ids, X = train_mod(list_rec)
    print("-->Predicting for given Ingredients")
    if args.ingredient:
        input_ing = [item for sublist in args.ingredient for item in sublist]
        input_ing  = ['þ' + 'þþ'.join(input_ing) + 'þ']
        #input_ing.append('þsea saltþ')
        with warnings.catch_warnings():
            warnings.filterwarnings(action = "ignore",category = FutureWarning)
            input_arg_mod = vectorize.transform(input_ing)
        X_in = input_arg_mod.toarray()
        #print(X_in)
        prediction = clf.predict(X_in)[0].upper()
        #Calculating the top-5 recipies:
        dist = DistanceMetric.get_metric('jaccard')
        dist_l  = dist.pairwise(X,X_in)
        dist_l = [item for sublist in dist_l for item in sublist]
        #print(dist_l)
        idx = np.argpartition(dist_l,5)
        index = idx[:5]
        ids_5 = [ids[i] for i in index]
        dist_5 = [dist_l[i] for i in index]
        print("-->Calculating distances")
        return prediction, ids_5, dist_5
         

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingredient", type=str, required = True, help = "Ingredient that you want to use.", nargs='*', action='append')
    global args
    args = parser.parse_args()    
    list_rec = get()
    prediction,ids_5,dist_5 = predict_cus(list_rec)        
    print("\t\tTop predicted Cuisine:",prediction)
    print("Top-5 closest Recipes:\n",list(zip(dist_5,ids_5)))

if __name__ == '__main__':
    main()
