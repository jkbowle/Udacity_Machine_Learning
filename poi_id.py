#!/usr/bin/python

import sys
import pickle
from sklearn import cross_validation
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression
#sys.path.append("../tools/")

from tester import test_classifier, dump_classifier_and_data
import text_results_to_dataset, get_sent_by_date

def scale_features(data, features_to_scale, all_features):
    ret_names = {}
    
    for feature in features_to_scale:
        ret_names[feature] = (scale_feature(data, feature))
        
    new_features = []
    for feature in all_features:
        if ret_names.get(feature,'NaN') != 'NaN':
            new_features.append(ret_names[feature])
        else:
            new_features.append(feature)
    
    return new_features
    

def scale_feature(data, feature_to_scale):
    new_name = feature_to_scale + "_scaled"
    scale_list = []
    keys = data.keys()
    for key in keys:
        val = data[key][feature_to_scale]
        if val != "NaN":
            scale_list.append(float(val))
    
    max_val = max(scale_list)
    for key in keys:
        val = data[key][feature_to_scale]
        if val == "NaN":
            data[key][new_name] = 'NaN'
        else:
            data[key][new_name] =  val/max_val
                    
    return new_name

def outlier_treatment(data, feature, elim_top=.1):
    print 'outlier.. treatment'
    chk_data = []
    for key in data.keys():
        chk_data.append(data[key][feature])
    
    chk_sort = sorted(chk_data, reverse=True)     
    num_elim = len(data)/elim_top
    
    elim_list = []
    
    for chk in chk_sort:
        if len(elim_list) <= num_elim:
            elim_list.append(chk)
        else:
            break
    print 'first to eliminate: '+str(chk_sort[0])
    for key in data.keys():
        value = data[key][feature]
        if value in elim_list:
            data[key][feature] = 'NaN'

def remove_key(in_dict,key):
    r = dict(in_dict)
    del r[key]
    return r      

def run_main():   
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    features_list = ['poi','email_subject','to_poi_ratio','combined', 'from_messages','expenses',
                     'deferred_income','other','restricted_stock'
                     ,'long_term_incentive','deferral_payments','email_body','restricted_stock_deferred'] # You will need to use more features
    
    ''' FEATURE LIST 
    bonus, deferral_payments, deferred_income, director_fees, email_address,
    email_body, email_subject, exercised_stock_options, expenses, from_messages, from_poi_to_this_person,
    from_this_person_to_poi, loan_advances, long_term_incentive, other, poi,
    restricted_stock, restricted_stock_preferred, salary, shared_receipt_with_poi,
    to_messages, total_payments, total_stock_value
        ------------ '''
    
    ### Load the dictionary containing the dataset
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    data_dict = remove_key(data_dict, 'TOTAL')
    #data_dict = pickle.load(open("my_dataset.pkl", "r") )
    get_sent_by_date.process_text_learning_features()
    data_dict = text_results_to_dataset.add_text_results(data_dict)
    
    def value_or_zero(inp):
        if inp == 'NaN':
            return 0
        else:
            return float(inp)
    ### Task 2: Remove outliers
    ### Task 3: Create new feature(s)
    # create percent email from poi
    for key in data_dict.keys():
        if data_dict[key]['to_messages'] == 'NaN':
            data_dict[key]['to_poi_ratio'] = 'NaN'
        else:
            data_dict[key]['to_poi_ratio'] = float(data_dict[key]['from_this_person_to_poi']) / float(data_dict[key]['from_messages'])
            
        combined = value_or_zero(data_dict[key]['salary']) + value_or_zero(data_dict[key]['bonus']) + \
            value_or_zero(data_dict[key]['total_stock_value']) + value_or_zero(data_dict[key]['total_payments']) + \
            value_or_zero(data_dict[key]['exercised_stock_options'])
        data_dict[key]['combined'] = combined
    # create percent email from poi
    ### Store to my_dataset for easy export below.
    features_list = scale_features(data_dict, features_list[1:], features_list)
    my_dataset = data_dict
    
    #outlier_treatment(my_dataset, 'combined', elim_top=.01)
    
    ### Extract features and labels from dataset for local testing
    #data = featureFormat(my_dataset, features_list, sort_keys = True)
    #labels, features = targetFeatureSplit(data)
    
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    
    #from sklearn.naive_bayes import GaussianNB
    #clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
    
    # Fit classifier with out-of-bag estimates
    from sklearn import ensemble
    params = {'n_estimators': 200, 'max_depth': 2,'min_samples_split':20,
              'learning_rate': .5, 'min_samples_leaf': 1}
    clf = ensemble.GradientBoostingClassifier(**params)
    
    #from sklearn.ensemble import AdaBoostClassifier
    #from sklearn.tree import DecisionTreeClassifier
    #clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2,min_samples_split=20),algorithm="SAMME",n_estimators=200)
    # RECALL: .39 features: 'email_subject','email_body','to_poi_ratio','combined' max_depth=3, min_samples_split=10
    # RECALL: .41 featuers  < SAME AS ABOVE but max_depth = 2
    # RECALL: .36 with just email_body & email_subject
    
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                     'C': [1, 10, 100, 1000]},
    #tuned_parameters = [{'C': [.001,1,.01,10]}]
    
    
    #from sklearn.grid_search import GridSearchCV
    #from sklearn.svm import LinearSVC
    #clf = GridSearchCV(LinearSVC(C=1,penalty="l2",class_weight='auto',loss="squared_hinge"), tuned_parameters, scoring='recall', verbose=3, n_jobs=5)
    
    
    
    # Maybe some original features where good, too?
    #fil = SelectKBest(f_regression, k=4)
    # create the pipeline to do the best selection:
    #clf = make_pipeline(fil, clf)
    #from sklearn.svm import LinearSVC      
    #clf = LinearSVC(C=.001,penalty="l2",class_weight='auto',loss="squared_hinge")
    
    
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script.
    ### Because of the small size of the dataset, the script uses stratified
    ### shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
    test_classifier(clf, my_dataset, features_list)
    weights = clf.feature_importances_
    for w, f in zip(weights,features_list[1:]):
        print str(w) + ' is the weight of '+f
        
    
    ### Dump your classifier, dataset, and features_list so 
    ### anyone can run/check your results.
    
    dump_classifier_and_data(clf, my_dataset, features_list)

if __name__ == '__main__':
    run_main()