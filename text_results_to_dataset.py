'''
Created on Oct 18, 2015

@author: ID19868
'''
import pickle
import numpy as np

def add_text_results(input_dict=None, pickle_dump=False):
    store =  pickle.load(open("author_dict.pkl", "r") )
    str_dict = store['dict']
    
    if not input_dict:
        input_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    
    for key in str_dict.keys():
        scores = str_dict[key]['text_score']
        s_scores = str_dict[key]['text_subj_score']
        input_dict[key]['email_body'] = np.median(scores)
        input_dict[key]['email_subject'] = np.median(s_scores)
        
    for key in input_dict.keys():
        input_dict[key]['email_body'] = input_dict[key].get('email_body','NaN')
        input_dict[key]['email_subject'] = input_dict[key].get('email_subject','NaN')
    
    if pickle_dump:
        pickle.dump(input_dict, open('my_dataset.pkl', "w") )
    return input_dict