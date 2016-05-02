#cosine similarity approach
#step1: predict labels based on cosine sim
from collections import defaultdict, OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from spacy import en
import re, math
from collections import Counter
import numpy as np

from datetime import datetime
import time
from datefinder import DateFinder


path = '/home/vagrant/datacourse/fellow_only_material/Capstone/'
train = []
for line in open(path + 'training.txt','r'):
    train.append(line[:-1])
    
real_events = []
for line in open(path + '2016042800-2016072800.txt'):
    real_events.append(line[:-1].split('::'))
    
class Sim_predictor():
    def __init__(self):
        self.weights = defaultdict(float)
        self.classes = []
        self.dates = []
        self.venues = []
        self.X = []
        self.WORD = re.compile(r'\w+')
        self.NUM = re.compile(r'\d+')


    
    #X is list of (str,YYYYMMDD)
    def fit(self, X, stop_words,K):
        self.dates = [z[1] for z in X]
        self.venues = [z[2] for z in X]
        self.classes = [z[0] for z in X]

        X = [' '.join(l) for l in X]
        self.X = X
        vectorizer = CountVectorizer()
        tfidf = TfidfTransformer()   
        #X = tfidf.fit_transform(vectorizer.fit_transform(train))
        #weights = {x[0]:x[1] for x in zip(vectorizer.get_feature_names(),tfidf.idf_)}
        w1 = {}
        w2 = {}
        for i,title in enumerate(self.classes):
            weight = (len(X) - i)**2
            for w in self.WORD.findall(title):
                w = w.lower()
                if w not in w1:
                    w1[w] = weight
                else:
                    w1[w] += weight
        Z = tfidf.fit_transform(vectorizer.fit_transform(self.classes))
        for w,idf in zip(vectorizer.get_feature_names(),tfidf.idf_):
            w2[w] = idf
        
        mean1, std1 = np.mean(w1.values()),np.std(w1.values())
        mean2, std2 = np.mean(w2.values()),np.std(w2.values())
        #popularity importance
        
        for k,v in w1.iteritems():
            self.weights[k] = K*math.exp((v-mean1)/std1)
        for k,v in w2.iteritems():
            self.weights[k] += (1-K)*math.exp((v-mean2)/std2)
            

            
        #dates are important
        for d in self.dates:
            self.weights[d] = 1
        #venues are important
        for v in self.venues:
            for w in v.split():
                self.weights[w] = 1
        for w in stop_words:
            self.weights[w] = 0.0
        
            
    def transform(self,X):
        return self
    
    def predict(self,X, threshold):
        dfin = DateFinder()
        pred_labels = []

        for s in X:
            s_dates = ' '.join('%.4d%.2d%.2d' % (dt.year,dt.month,dt.day) for dt in dfin.find_dates(s))
            date_str = dfin.extract_date_strings(s)
            #print s_dates
            for st in date_str:
                #print st[0],
                s = s.replace(st[0],'')
            s = ' '.join([s,s_dates])
            m = (None,0)
            #print s.lower()
            for i, label in enumerate(self.X):
                #date = self.dates[i]
                
                d1 = self.get_cosine(s.lower(), label.lower(),self.weights,threshold)
                #print d1,label
                #d2 = max(1,(date in s_dates)*2)
                if d1 > m[1]:
                    m = (label, d1)
            pred_labels.append((s,m))
        return pred_labels
        
    def get_cosine(self, text1, text2, weights, threshold):
        vec1 = self.WORD.findall(text1)
        vec2 = self.WORD.findall(text2)

        w1 = self.sumW(vec1,self.weights)
        w2 = self.sumW(vec2,self.weights)
        denominator = math.sqrt(w1*w2)
        if w1 < w2:
            if w1 < denominator*threshold: return 0.0
        elif w2 < denominator*threshold: return 0.0
        if not denominator: return 0.0

        intersection = set(vec1) & set(vec2)
        numerator = self.sumW(intersection, self.weights)
        # if only dates match, ignore
        if len(intersection) == 1:
            if self.NUM.search(list(intersection)[0]):
                return 0.0
        else:
            return float(numerator) / denominator

    def sumW(self, vec, weights):
        s = 0.0
        for w in vec:
            #ignore numbers
            #if not self.NUM.match(w):
            s += self.weights.get(w,0.0001)
        return s

    def get_overlap(self, text1, text2, weights):
        vec1 = self.WORD.findall(text1)
        vec2 = self.WORD.findall(text2)
        intersection = set(vec1) & set(vec2)
        return self.sumW(intersection,weights)

def extract():
    #from spacy.en import English
    #nlp = English()
    print 'quit to close.'
    threshold = 0.1
    model = Sim_predictor()
    print 'Fitting model...'
    sw = en.STOPWORDS.union(set(['ticket','tickets','san','francisco','s']))
    model.fit(real_events, sw,0.01)
    spl = re.compile(r'(.*)\s+(\d{8})\s+(.*)')
    while True:
        text = raw_input('Text to parse: ')
        if text == 'quit': break
        to_print = model.predict([text],threshold)[0][1][0]
        if to_print:
            info = spl.match(to_print)
            if info:
                g = info.groups()
                print 'What: ' + g[0]
                print 'When: ' + datetime.strptime(g[1],'%Y%m%d').strftime("%A, %d %B %Y")
                print 'Where: ' + g[2]
            else:
                print 'Not found'
        else:
            print 'Not found'
      
if __name__=='__main__':
    extract()
