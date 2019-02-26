# Cuisine Prediction and 5 Closest dishes Recommendation
Used Jaccard's distance to recommend closest 5 dished based of ingredients input through command line args. predicted closest cuisine of the same.(Written May 2018)


#======================================Project 2: =========================================#
FILE Organization:

	project2/
		project2/
			__init__.py
			project2.py
		tests/
			test_eda.py
			test_train_mod.py
		yummly.json
		README
		requirements.txt
		setup.cfg
		setup.py

#==============================Predict Cuisine and Top-5 closest Recipies=================#
Execution Instructions:
- The project2.py is run from this directory: 
					projects/project2
-The following command can be used to execute the py file correctly:

	python3 project2/project2.py --ingredient 'wheat' --ingredient 'oil' --ingredient 'salt' --ingredient 'water'

------------------------------------------------------------------------------------------------------
We are given the "yummly.JSON" file which when loaded using the JSON package gives us a list of 
dictionaries with the following format:

[{'id':1 , 'cuisine' : 'martian', 'ingredients' : ['iron','nails','magnets']},{'id':2},....,{}]

General Approach:
We first extract the contents of the list of dictionary to individual lists. The ingredients list is vectorized using 
thetfidf and also the CountVectorizer. The output is used to train the different types of models to predict the
target variable "Cuisine". Once we get a predcicted cuisine we het the top_5 closeest recipies with the ingredients 
that we have listed using different types of distances.

The predicted cusine and the top 5 recipies are displayed along with their distnces from our datapoint.


===============Methods==============
-eda(): eda() is mainly used to create individual lists.
	In the method eda we perform some basic exploratory data analysis trying to understand things like number of 
	classes (20), number of dataoints(~39700), number of unique ingredients(~6100) etc.
	
	We se this space to extract the 'id's, cuisines and ingredients from the list of dictioanries.

-train_mod():This method takes in the list of dicts calls eda(). eda() returns y(cuisine list),ids list,and thelist of list of ingredients.
	As CountVectorizer and TfidfVectroizer require input in the form of list of strings. One string for one document. But here we 
	have already vectroized strings.Hence the ingredients are joined using the thorn character as a sperator. Another paramter is the 
	min_df. When we give a float value, the float part is used as a minmum threshold to keep or throw away ingredients. 
	So here the 0.0001 says that only phrases that have atleast appeard in 0.01% of all the documents are retained.
	Here Count if performing better than tfidf. This tells us that the ingredients that are rpeated in different recipies 
	are still important to find patterns int he data.

	This is done as both the vectorizers have a paramter called token_pattern. usng this we can manipulate the defintion of 
	a token and what is vectorized. r'thorn(.*?)thorn' is the pattern used to seperate the new string to get the original ingreidents.
	We obtain the matrix X from the SRC matric and then split the data into train and test sets for comparison of differnt models.
	We used the following classifiers for the task:
   a) Multinomial Naive Bayes b)Bernoulli Naive bayes c)MLP Classfier(Dense Nueral Network)
	MLP and Multinomial perform wth similar test accuracy scores. The latter takes much lesser time to train and hence is preffered.
	
	Finally this method returns the model, vectorizer, ids, and the feature matrix.

-predict_cus(): predict_cus() method predicts the cuisine and the top-5 recipies.
	
	train_mod() is called to obtain its contents.The vectorizer is used to tranform the new user input and then is fed to transform() method 
	of the classifier. We obtain the predicted class this way. For we the top-5 ids, teh distnace metric is used to calculate the 
	distance between all the datapoints and the user input. The closest points are then taken and displayed. For caluclating the distance,
	various distances were tried such as 'hamming','dice' and 'jaccard'. We finally used jaccard.
	
	This method returns the cuisine PREDICTION and the ids and their distances.
