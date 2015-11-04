#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string


cachedStopWords = stopwords.words("english")

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        #words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        
        word_list = []
        words = text_string.split()
        stemmer = SnowballStemmer("english")
        
        for word in words:
            stem = stemmer.stem(word)
            if len(stem) > 0:
                word_list.append(stem)

        words = " ".join(word_list)

    return words 

def stem_words(email_text):
    word_list = []
    #words = email_text.split()
    words = ' '.join([word for word in email_text.split() if word not in cachedStopWords])
    words = words.split()
    stemmer = SnowballStemmer("english")
        
    for word in words:
        stem = stemmer.stem(word)
        if len(stem) > 0:
            word_list.append(stem)
    words = " ".join(word_list)
    return words

def full_email_parse(f, stem=True, header=True, stem_subject=False):
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    adict = {}
    if len(content) > 1:
        ### remove punctuation
        if stem:
            text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
            words = stem_words(text_string)
        
        if header:
            adict = parse_attributes(content[0])
            if stem_subject:
                adict['Subject'] = stem_words(adict.get('Subject',''))
    
    return words, adict

def parse_attributes(email_header):
    import re
    header_dict = {}
    pattern = re.compile('((?:^|(?:\.\s))(\w+)((-\w+)+: |(: )))')
    lines = email_header.splitlines()
    key = None
    for line in lines:
        m = pattern.match(line)
        if m:
            key = m.group(0)[:-2]
            value = line[len(m.group(0)):]
            header_dict[key] = value.strip()
        else:
            header_dict[key] = header_dict[key] + " " + line.strip()
        
    return header_dict
    
def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()

