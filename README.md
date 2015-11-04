# Udacity_Machine_Learning
Class files for the Intro to Machine Learning class on udacity.com

## explanation for each of the files on this account

### author_dict.pkl
This is the pickle file for the dictionary that stores the text mining results (after removing stop words and doing some stemming)
It also stores the text learning score before it is added to the final dataset
format {'dict':{employee_name:{'text_score':x.xxx,'text_subj_score':x.xxx}, etc..},'all':{employee_name:[[doc1.words,etc.],[doc2.words,etc..],etc.],etc.}}

### clf_results.txt
Text file where I stored most of my runs for each iteration of the model

### EnronMachineLearning_JasonBowles.pdf
Final write up to answer the questions for the project

### get_sent_by_date.py
Python script to iterate through emails and get the data ready for the text learning portion of this model.  (note that the paths need to be updated in order to run another machine (and the ud120 project structure is assumed).  The parsing is completed and stored in the author_dict.pkl file in case this step is desired to be skipped.

The second part of this script focuses on the text learning portion of the full model.  It creates the vectorizer and classifier and then runs it and pulls out the decision_fuction and stores that in author_dict.pkl

### poi_email_addresses.py
Hard coded list of all valid poi email addresses

### final_project_dataset.pkl
The initial dataset provided by the course and udacity

### my_classifier.pkl
The final classifier for project.  GradientBoostingClassifier

### my_dataset.pkl
The final dataset used to run against the classifier

### poi_id.py
The modified version of the given poi_id.  It has code in it to call out to get_sent_by_date.py and text_results_to_dataset.py and then create the final dataset through feature scaling, engineering and algorithm creation

### README.md
this file

### tester.py
The given tester python script that uses the StratifiedShuffleSplit cross validation strategy to create 1000 k folds and then count the number True Positives, True Negatives, False Positives and False Negatives and then give the final score (Accuracy, Precision, Recall, F1 and F2)

### test_results_to_dataset.py
Stores the text_learning results to the final dataset

### websites_used.txt
The list of websites used during model creation and final write up.

### tools folder
The provided udacity tools scripts to auto run, assess and store your model
