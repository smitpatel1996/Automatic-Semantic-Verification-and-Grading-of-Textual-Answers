import re
import pickle
import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

def penn_to_wn(tag):
	if tag.startswith('N'):
		return 'n'
	if tag.startswith('V'):
		return 'v'
	if tag.startswith('J'):
		return 'a'
	if tag.startswith('R'):
		return 'r'
	return None

def tagged_to_synset(word, tag):
	wn_tag = penn_to_wn(tag)
	if wn_tag is None:
		return None
	try:
		return wn.synsets(word, wn_tag)[0]
	except:
		return None

ans = "To simulate the behaviour of portions of the desired software product."
mSynset = []
dum = re.sub('\W+',' ',ans)
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(dum)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
s = pos_tag(filtered_sentence)
synset = [tagged_to_synset(*tagged_word) for tagged_word in s]
mSynset = [ss for ss in synset if ss]

def sentence_similarity(synset,sentence,b):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    sentence = pos_tag(filtered_sentence)
    synsets = [tagged_to_synset(*tagged_word) for tagged_word in sentence]
    synsets = [ss for ss in synsets if ss]	
    score, count = 0.0, 0
    fin = []
    for i in synset:
        a = [i.path_similarity(ss) for ss in synsets]
        a = [ss for ss in a if ss]
        if(len(a) == 0):
        	best_score = 0.0
        else:	
        	best_score = max(a)
        score += best_score
        count += 1
        fin.append(best_score)	
    score /= count
    return fin

def round_off(x):
	q = x-int(x)
	if(q < 0.25):
		return int(x)
	elif(q > 0.25 and q < 0.75):
		return round(int(x)+0.5,1)
	else:
		return int(x)+1	

max_marks = 5
df = pd.read_csv("dtst1.csv")
X = np.array(df[['Answer']])
y = np.array(df[['Marks']])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.60)

X_train = X_train.tolist()
y_train = y_train.tolist()
print y_train
model_X = []
model_y = []
for i in range(0,len(X_train)):
	if(y_train[i][0] == max_marks):
		o = []
		dum = re.sub('\W+',' ',X_train[i][0])
		stop_words = set(stopwords.words('english'))
		word_tokens = word_tokenize(dum)
		filtered_sentence = [w for w in word_tokens if not w in stop_words]
		s = pos_tag(filtered_sentence)
		synset = [tagged_to_synset(*tagged_word) for tagged_word in s]
		o = [ss for ss in synset if ss]
		mSynset = mSynset+o

for i in range(0,len(X_train)):
	fin = sentence_similarity(mSynset,X_train[i][0],False)	
	model_X.append(fin)
	model_y.append(y_train[i][0]/max_marks)

reg1 = linear_model.Ridge()
reg1.fit(model_X, model_y)
print "Ridge training done"
joblib.dump(reg1, 'Ridge.pkl')

X_test = X_test.tolist()
y_test = y_test.tolist()
eRidge = 0
for i in range(0,len(X_test)):
    fin_predict = sentence_similarity(mSynset,X_test[i][0],False)
    predicted = reg1.predict([fin_predict])
    final = predicted[0]*max_marks
    if(final > max_marks):
    	final = max_marks
    print "Predicted: " + str(round_off(round(final,1)))+ " Actual: " + str(y_test[i][0])
    eRidge = eRidge + (round_off(round(final,1))-y_test[i][0])**2

print "Ridge RMS error: " + str((eRidge/len(X_test))**0.5)
