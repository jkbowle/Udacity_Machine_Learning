'''
Created on Oct 13, 2015

@author: ID19868
'''

from tools import parse_out_email_text
from dateutil.parser import parse
from poi_email_addresses import poiEmails
import os
import csv
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tools.feature_format import textFormat
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn import svm
import operator
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import gc


def get_text_from_emails(emails, email_key, poi_emails, data_dict,from_pkl=False, combined=False):
    if from_pkl:
        try:
            store =  pickle.load(open("author_dict.pkl", "r") )
            str_combined = store['all']
            str_dict = store['dict']
            if combined != str_combined:
                print 'stored processing of emails does not match arguments.. re-processing emails'
                return process_emails(emails, email_key, poi_emails, data_dict,all_together=combined)
            else:
                return str_dict
        except IOError:
            return process_emails(emails, email_key, poi_emails, data_dict,all_together=combined)
    return process_emails(emails, email_key,poi_emails, data_dict, all_together=combined)
    

def process_emails(emails, email_key, poi_emails, data_dict,num_process=-1, all_together=True):
    processed = 0
    old_text = 'enron_mail_20110402'
    new_text = r'C:\Users\id19868\Documents\Udacity\ud120-projects'
    direct = r'C:\Users\id19868\Documents\Udacity\ud120-projects\final_project\emails_by_address'
    
    files = os.listdir(direct)
    
    emails_from_all = []
        
    csvfile = open('email_subjects.csv','w')
    writer = csv.DictWriter(csvfile, ['id','email_address','subject','sent_date', 'poi_ind','file_name'],lineterminator='\n')
    writer.writeheader()
    email_list = r'C:\Users\id19868\Documents\Udacity\ud120-projects\final_project\emails_by_address\from_adam.umanoff@enron.com.txt'
    for f in files:
        email_list = os.path.join(direct,f)
        email_address = 'someone@somewhere.com'
        if f.startswith("from_"):
            email_address = f[len('from_'):-4]
        
        if email_address in emails and (num_process < 0 or processed < num_process):
            print 'processing: '+email_address
            one_file = None
            with open(email_list) as filelist:
                for line in filelist:
                    one_file = line
                    one_file = one_file.replace("/","\\")
                    one_file = one_file.replace(".","")
                    one_file = one_file.replace("\n","")
            
                    new_file = one_file.replace(old_text, new_text)
            
                    with open(new_file) as email:
                        body, header_dict = parse_out_email_text.full_email_parse(email, stem=True, header=True, stem_subject = True)
                        from_date = parse(header_dict['Date'])
                        subject = header_dict['Subject']
                        poi_ind = "Y" if email_address in poi_emails else "N"
                        from_date = from_date.strftime('%m/%d/%Y %H:%M:%S.%f')
                        _id = email_address+"_"+subject+"_"+from_date+"_"+body[:100]
                        dict_row = {'id':_id,'email_address':email_address,'subject':subject, 'sent_date':from_date,'poi_ind':poi_ind,'body':body,'file_name':new_file}
                        
                        #writer.writerow(dict_row)
                        emails_from_all.append(dict_row)
                        processed = processed + 1
                    
    
    data_sort = sorted(emails_from_all, key=lambda entry: entry['id'])
    new_list = []
    print 'now sorting the results and eliminating duplicates'
    # first_100_chars
    lst_id = None
    for entry in data_sort:
        _id = entry['id']
        if _id != lst_id:
            writer.writerow({'id':entry['id'],'email_address':entry['email_address'],'subject':entry['subject'], 'sent_date':entry['sent_date'],'poi_ind':entry['poi_ind'],'file_name':entry['file_name']})
            new_list.append(entry)
        
        lst_id = _id
    #new_file = r'C:\Users\id19868\Documents\Udacity\ud120-projects\maildir\blair-l\_sent_mail\1'
    
    scrub_words = ()
    
    author_dict = {}
    last_email = None
    print 'now pulling the subject and body by author'
    for entry in new_list:
        words = entry['body']
        subject = entry['subject']
        email = entry['email_address']
        key = email_key[email]
        if email != last_email:
            author_dict[key] = {'body':[],'subject':[], 'poi':1 if data_dict[key]['poi'] else 0}
            if all_together:
                author_dict[key] = {'body':'','subject':'', 'poi':1 if data_dict[key]['poi'] else 0}
            
            
            last_email = email
            
        for scrub in scrub_words:
            words = words.replace(scrub, "")
        
        for scrub in scrub_words:
            subject = subject.replace(scrub, "")
        
        if all_together:
            author_dict[key]['body'] = author_dict[key]['body'] + ' ' + words
            author_dict[key]['subject'] = author_dict[key]['subject'] + ' ' + subject
        else:
            author_dict[key]['body'].append(words)
            author_dict[key]['subject'].append(subject)
        
    
    
    pickle.dump({'all':all_together,'dict':author_dict}, open('author_dict.pkl', "w") )
    return author_dict

def scrub_text(dictionary, features, scrub_list):
    word_list = False
    try:
        dictionary[dictionary.keys()[0]][features[0]].replace('try','try')
    except AttributeError:
        word_list = True
    for key in dictionary.keys():
        for feature in features:
            words = dictionary[key][feature]
            for scrub in scrub_list:
                if word_list:
                    new_list = []
                    for word in words:
                        word = word.replace(scrub,"")
                        new_list.append(word)
                    words = new_list
                else:
                    words = words.replace(scrub, "")
            
            dictionary[key][feature] = words

def vectorize_text_first(author_dict, feature, scrub_list, test_size = 0.1, all_together=False):
    vectorizer, new_scrub, recursion, workset =vectorize_text(author_dict, feature, scrub_list, test_size, all_together)
    
    if recursion:
        print 'calling vectorize AGAIN!!!'+str(len(new_scrub))
        vectorizer = None
        gc.collect()
        vectorize_text_first(author_dict, feature, new_scrub, test_size)
    
    return vectorizer, new_scrub, recursion, workset

def vectorize_text(author_dict, feature, scrub_list, test_size = 0.1, all_together=False):
    scrub_text(author_dict, [feature], scrub_list)

    labels, features, authors = textFormat(author_dict,'poi',feature, sort_keys=True, all_together=all_together)
    #labels, features = targetFeatureSplit(data)
    
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=test_size, random_state=42)
    print 'now vectorizing'
    
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
    features_train = vectorizer.fit_transform(features_train)
    features_test  = vectorizer.transform(features_test).toarray()
    
    names = vectorizer.get_feature_names()
    
    use_dual = True
    if feature == 'subject':
        use_dual=False 
    
    #clf = tree.DecisionTreeClassifier(min_samples_split=4,min_samples_leaf=4)
    clf = LinearSVC(C=.001,penalty="l2",dual=use_dual,class_weight='auto',loss="squared_hinge")
    print 'about to train the model: trying to identify: '+str(sum(labels_train))
    print '    number of features using for this model: '+str(len(names))
    clf.fit(features_train,labels_train)
    print '     now making a prediction'
    prediction = clf.predict(features_test)
    print '    calculating the accuracy'
    accuracy = accuracy_score(labels_test, prediction)
    print '    accuracy: ' +str(accuracy)
    
    recall = recall_score(labels_test,prediction)
    print '---------> SKLEARN calcs: (Precision: '+str(precision_score(labels_test, prediction))+', Recall: '+str(recall)+")"
    
    print 'now checking acuracy against training'
    prediction = clf.predict(features_train)
    print '    calculating the accuracy'
    accuracy = accuracy_score(labels_train, prediction)
    print '    accuracy: ' +str(accuracy)
    
    ### your code goes here
    
    weights = None
    try:
        weights = clf.feature_importances_
    except AttributeError:
        weights = clf.coef_
    
    
    high_weights2 = []
    high_weights = []
    length = 0
    try: 
        length = len(weights[0])
    except TypeError:
        length = len(weights)
        weights = [weights]
    
    match = max(weights[0])
    for i in range(length):
        weight = weights[0][i]
        if weight >= match:
            high_weights2.append({'num':i,'weight':weight})
    
    high_weights2 = sorted(high_weights2, key=lambda entry: entry['weight'],reverse=True)
    
    length = 10 if len(high_weights2) > 10 else len(high_weights2)
    
    for i in range(length):
        high_weights.append(high_weights2[i])
        
    recursion = False
    for i in range(len(high_weights)):
        weight = high_weights[i]['weight']
        print '        The heighest weight: '+str(weight)
        print '        The subscript of that weight: '+str(high_weights[i]['num'])
        print '        name of that weight ---------------> '+str(names[high_weights[i]['num']])
        if recall < 0.70 or weight > 1:
            recursion = True
            new_scrub = scrub_list
            new_scrub.append(names[high_weights[i]['num']])
            
            return vectorizer, new_scrub, recursion, [labels, clf.decision_function(vectorizer.transform(features)), authors]
    
            
    return vectorizer, scrub_list, recursion,  [labels, clf.decision_function(vectorizer.transform(features)), authors]

def look_for_common_words(in_dict, feature, data_dict, poi_only=True, top=10):
    top_ = []
    for key in in_dict.keys():
        poi_ind = data_dict[key]['poi']
        
        if poi_only and poi_ind:
            word_cnt = {}
            words = in_dict[key][feature]
            word_list = words.split()
            for word in word_list:
                word_cnt[word] = word_cnt.get(word,0) + 1
            sorted_x = sorted(word_cnt.items(), key=operator.itemgetter(1),reverse=True)
            for i in range(top):
                top_.append(sorted_x[i][0])
    
    print 'got the top '+ str(top) + ' of each'
    print top_
    
def process_text_learning_features():
    poi_emails = poiEmails()
    _all = False
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    emls = []
    eml_key = {}
    for key in data_dict.keys():
        emls.append(data_dict[key]['email_address'])
        eml_key[data_dict[key]['email_address']] = key
    
    author_dict = get_text_from_emails(emls,eml_key,poi_emails, data_dict,from_pkl=True, combined=_all)
    print 'Got the Author Text Dictionary!!!'
    
    #look_for_common_words(author_dict, 'body', data_dict)
    subj_scrub = []
    body_scrub = ['ddelainnsf','delainey','regard','david','houect','delaineyhouect','christoph', 'jeff', 'product', 'kitchen', 
                  'jdasovicnsf','allegheni','pastoria','jdasovicnsf','neuner','jacobi','catalytica','calpin',
                  'ect','dave','tim','ena','62602pst','belden','guy','chris','calger','valu','salisburi',
                  'swerzbin','kelli','paula','motley','chapman','johnson','frevertna','presto','ben','ray','janet','wes','dietrich',
                  'deal','holden','kay','floor','thxs','portland','manag','plan','turbin','enron','board','meet','forward','year',
                  'term','pdx','market','goal','lavoratocorp','2702pst','desk','unit','discuss','mid','2000','kellycom','7138532534',
                  '7138536485','execut','power','cost','busi','complet','ensur','howev','provid','sheet','short','right','structur',
                  'trade','organ','peopl','jskillinpst','corp','generat','kevin','dash','rob','view','sale','need']
    #body_scrub = ['ddelainnsf','delaineyhouect','hang','court','advis','delainey','david'
    #              'ect','0921','ray','todaytonight','lavoratoenron','let'
    #              ]
    #body_scrub = ['pastoria','calpin','amita','ecc','turbin','las','calgerpdxectect','calgerpdxect','allegheni', 
    #'dwr','catalytica','cdwr','2mm','qf','02062001','vega','creditworthi','psco','calger', 
    #'5mm','jacobi','erc','01262000','7mm','10mm','jdasovicnsf','pcg','parquet','goldendal', 
    #'ae','eix','neuner','4mm','5034643735','helpdesk','christoph','louis', 'product', 'kitchen', 
    #'christoph', 'jeff', 'david', 'delainey', 'ben']
    s_vectorizer, scrubs_subj, recur, workset = vectorize_text_first(author_dict, 'subject',subj_scrub,test_size=0.4, all_together=_all)
    store_results(workset, author_dict, 'text_subj_score')
    b_vectorizer, scrubs_body, recur, workset = vectorize_text_first(author_dict, 'body',body_scrub,test_size=0.4, all_together=_all)
    store_results(workset, author_dict, 'text_score')
    print scrubs_subj
    print scrubs_body
    pickle.dump({'all':_all,'dict':author_dict}, open('author_dict.pkl', "w") )

def store_results(workset, author_dict, feature_name):    
    authors = workset[2]
    features = workset[1]
    print 'storing results'
    for key in author_dict.keys():
        author_dict[key][feature_name] = []
    for i in range(len(authors)):
        author = authors[i]
        feature = features[i]
        author_dict[author][feature_name].append(feature)
    
    

