# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 21:16:56 2017

@author: Ashish Kumar Jayant
@title: Hackerearth sentiment analysis problem : Predict Happiness Challenge 
"""

from sklearn.externals import joblib
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
import re
import nltk
import time
import sys
from textblob import TextBlob


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB







class functions:
    def __init__(self):
        pass
    #-----------------------------------------------------------------------------------------------------------------
    #           TEXT PREPROCSSING - CLEANING,NUMERICAL REPRESENTATION OF TEXT USING COUNT VECTORIZER
    #-----------------------------------------------------------------------------------------------------------------
    def clean_text(self,text,stops,lowercase=False,stop=False,lemma=False):
        txt = str(text)
        txt = re.sub(r'[^A-Za-z0-9\s]',r'',txt)
        txt = re.sub(r'\n',r' ',txt)
        if lowercase:
            txt = " ".join([w.lower() for w in txt.split(" ")])
        if stop:
            txt = " ".join([w for w in txt.split(" ") if w not in stops])
        if lemma: 
            st =  nltk.stem.SnowballStemmer('english')
            txt = " ".join([st.stem(w) for w in txt.split(" ")]) 
        return txt
    def to_labels(self,x):
        if x[0][0]<x[0][1]: 
            return "happy"
        else:
            return "not_happy"
        
    def initialise(self):
        train_df = pd.read_csv("D:\\New folder\\f2c2f440-8-dataset_he\\train.csv")
        test_df = pd.read_csv("D:\\New folder\\f2c2f440-8-dataset_he\\test.csv")    
        #test_df.loc[-1]=[test_case.get("user_id"),test_case.get("txt"),test_case.get("browser"),test_case.get("device")]
        #test_df.index=test_df.index+1
        test_df['Is_Response'] = np.nan
        combined = pd.concat([train_df,test_df]).reset_index(drop=True)
        stops = set(stopwords.words("english"))
        combined['Description'] = combined['Description'].map(lambda x: functions.clean_text(self,x,stops,lowercase=True,stop=True,lemma=True))
        count_fit = CountVectorizer(analyzer='word',ngram_range=(1,3),min_df=95,max_features=3750)
        count_data = count_fit.fit_transform(combined['Description'])
        cols = ['Browser_Used','Device_Used']
        for x in cols:
            lbl = LabelEncoder()
            combined[x] = lbl.fit_transform(combined[x])
        count_df = pd.DataFrame(count_data.todense())
        count_df.columns = ['col' + str(x) for x in count_df.columns]   
        combined['Length'] = combined['Description'].apply(len)
        count_train = count_df[:len(train_df)]
        count_test = count_df[len(train_df):]
        train_feats = combined[~pd.isnull(combined.Is_Response)]
        test_feats = combined[pd.isnull(combined.Is_Response)]
        train_feats['Is_Response'] = [1 if x == 'happy' else 0 for x in train_feats['Is_Response']]
        cols.append('Length')
        train_feats2 = pd.concat([train_feats[cols],count_train],axis=1)
        test_feats2 = pd.concat([test_feats[cols],count_test],axis=1)
        target = train_feats['Is_Response']
        print("Vectors created")
        return train_feats2,target,test_feats2,count_fit,test_df
    
    def make_feature(self,size,txt):#,browser,device):
        import numpy as np
        import nltk
        x=np.zeros([size])
        st =  nltk.stem.SnowballStemmer('english')
        stops = set(stopwords.words("english"))
        txt = " ".join([w for w in txt.split(" ") if w not in stops])
        txt = " ".join([w.lower() for w in txt.split(" ")])
        txt = " ".join([st.stem(w) for w in txt.split(" ")]) 
        
        
        x[0]=2
        x[1]=1
        x[2]=len(txt)
        
        vocab=count_fit.get_feature_names()
        for i in txt.split(" "):
            try:
                ind =  vocab.index(i)
                if x[ind+3] == 0:
                    x[ind+3] =1
                else:
                    x[ind+3] +=1
            except:
                pass
        x=x.reshape(1,-1)
        print("Feature vector created")
        return x    
    
    #----------------------------------------------
    # PICKLING OF MODELS
    #-----------------------------------------------
            
    def test_sentiment(self,txt):#,browser,device):
        x=functions.make_feature(self,feats,txt)#,browser,device)
        return functions.to_labels(self,eclf.predict_proba(x))   
    def save_pickle(self):
        joblib.dump(eclf,'my_ensemble.pkl') 
        joblib.dump(feats,'feats.pkl')
        joblib.dump(count_fit,'vectorizer.pkl')
        print("pickle saved")
    def load_pickle(self):
        files = ["my_ensemble.pkl","feats.pkl","vectorizer.pkl"]      
        eclf = joblib.load(files[0])
        feats = joblib.load(files[1])
        count_fit = joblib.load(files[2])
        
        return eclf,feats,count_fit
    #----------------------------------------------------------------------------------------------
    #       MACHINE LEARNING MODELS
    #---------------------------------------------------------------------------------------------
    def make_model(self):
        #---------------------------------------------------------------------------------------------
        #                       TREE BASED ALGORITHMS
        #---------------------------------------------------------------------------------------------
        
        #--Chossing random_state parameter
        #------Basically, a sub-optimal greedy algorithm is repeated a number of times using----------
        #------random selections of features and samples (a similar technique used in random----------
        #------ forests).The 'random_state' parameter allows controlling these random choices---------
        
        #--n_estimators = no of decision trees to be created in forest
        
        model_rf = RandomForestClassifier(n_estimators=145,random_state=10,n_jobs=-1)
        model_rf.fit(train_feats2,target)
        
        model_gb = GradientBoostingClassifier(n_estimators=145,random_state=11,n_jobs=-1)
        model_gb.fit(train_feats2,target)
        
        model_ab = AdaBoostClassifier(n_estimators=145,random_state=12,n_jobs=-1)
        model_ab.fit(train_feats2,target)
        
        
        #--------------------------------------------------------------------------------------------
        #               LOGISTIC REGRESSION
        #--------------------------------------------------------------------------------------------
        
        model_lr = LogisticRegression(random_state=1)
        model_lr.fit(train_feats2,target)
        
        #--------------------------------------------------------------------------------------------
        #               NAIVE BAYES
        #--------------------------------------------------------------------------------------------
        
        model_nb = MultinomialNB()
        model_nb.fit(train_feats2,target)
        
        #--------------------------------------------------------------------------------------------
        #               VOTING ENSEMBLE OF ALL MODELS
        #--------------------------------------------------------------------------------------------
        

        
        clf = [model_rf,model_lr,model_gb,model_ab,model_nb]
        eclf = EnsembleVoteClassifier(clfs=clf, weights=[1,2,1,1,1] ,refit=False)   #weights can be decided by stacking!!
        eclf.fit(train_feats2,target)
        print("model created")
        preds = eclf.predict(test_feats2)
        sub3 = pd.DataFrame({'User_ID':test_df.User_ID, 'Is_Response':preds})
        sub3['Is_Response'] = sub3['Is_Response'].map(lambda x: functions.to_labels(self,x))
        sub3 = sub3[['User_ID','Is_Response']]
        sub3.to_csv('D:\\New folder\\f2c2f440-8-dataset_he\\SUB_TEST.csv', index=False)
        print("prediction saved")
        return eclf
    
'''
#--------------DRIVER---------------------------------------------   
f=functions() 
train_feats2,target,test_feats2,count_fit,test_df=f.initialise()
feats = train_feats2.shape[1]
eclf = f.make_model()
f.save_pickle()
'''

#--running second time    
f=functions()
eclf,feats,count_fit = f.load_pickle()

#--------------------------------------------------------------------------------------------------
# I combined in-built textblob sentiment analyzer with my own trained model to assign confidence
#--------------------------------------------------------------------------------------------------
while(True):
    tx=input('enter the text -> ')        
    a=f.test_sentiment(tx)
    b_score=TextBlob(tx).sentiment.polarity
    b=["happy" if b_score>0 else "not_happy"]
    if a==b[0]:
        print(a,' |confidence: high')
    else:
        if b_score>1 or b_score<-1:
            print(b[0],'|confidence: low')
        else:
            print(a,'|confidence: low')





#-------------ALL TESTING STUFF BELOW----NOT PART OF CODE-------------

#-----------------------------------------------------------------------------------------------------------------------
#   TESTING VOTING ENSEMBLE OF RANDOM FOREST, LOGISTIC REGRESSION, GRADIENT BOOST,ADABOOST, GAUSSIAN NB
#-----------------------------------------------------------------------------------------------------------------------
'''
# ------------------Algorithms performance analysis for weights of ensemble(ONLY FOR TESTING)-------------------------
x_train,x_test, y_train,y_test = train_test_split(train_feats2,target,test_size=0.2,random_state=0)


model_rf = RandomForestClassifier(n_estimators=150,random_state=1)
model_rf.fit(x_train,y_train)


model_lr = LogisticRegression(random_state=11)
model_lr.fit(x_train,y_train)

model_gb = GradientBoostingClassifier(n_estimators = 100,random_state=1)
model_gb.fit(x_train,y_train)

model_ab = AdaBoostClassifier(n_estimators =100,random_state=11)
model_ab.fit(x_train,y_train)

clf = [model_rf,model_lr,model_gb,model_ab]
score = []
for c in clf:
    pred = c.predict(x_test)
    score.append(accuracy_score(y_test,pred,normalize=True))
    print(str(c).split("(" )[0],accuracy_score(y_test,pred,normalize=True))

#---------------------------------------------------------------------------------------------
#-------Logistic Regression performed best and rest of them were comparatively similar-------- 
#-------so I gave model_lr weight of 2 and rest of them 1-------------------------------------
#---------------------------------------------------------------------------------------------

eclf = EnsembleVoteClassifier(clfs=clf, weights=[1,2,1,1,1] ,refit=False)
eclf.fit(x_train,y_train)
preds = eclf.predict(x_test)
ensemble_score = accuracy_score(y_test,preds)
print("ensemble score - ",ensemble_score)


#---------------------------------------------------------------------------------------------------
# ACCURACY - 88.583 % ON PUBLIC LEADERBOARD
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
#--     DEEP LEARNING :  FEEDFORWARD NEURAL NETWORK (*KERAS FRAMEWORK) : just a trial!
#---------------------------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense
#from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

#-Normalizing the data--
y_binary = to_categorical(target)
scaler = MinMaxScaler()
train_feats2=scaler.fit_transform(train_feats2)
test_feats2=scaler.fit_transform(test_feats2)
train_feats2=np.array(train_feats2)
test_feats2=np.array(test_feats2)

#-Building model--
model=Sequential()
model.add(Dense(2000, activation='relu', input_shape=(5616,)))
model.add(Dense(2000, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics = ['accuracy'])

history = model.fit(train_feats2, y_binary, 
          batch_size=32, 
          epochs=1, 
          validation_split = .2,
          verbose=2)

#-Saving predictions--
preds = model.predict_classes(test_feats2)
sub3 = pd.DataFrame({'User_ID':test_df.User_ID, 'Is_Response':preds})
sub3['Is_Response'] = sub3['Is_Response'].map(lambda x: to_labels(x))
sub3 = sub3[['User_ID','Is_Response']]
sub3.to_csv('D:\\New folder\\f2c2f440-8-dataset_he\\SUB_TEST.csv', index=False)

#---------------------------------------------------------------------------------------------
# ACCURACY  87.772 % ON PUBLIC LEADERBOARD
# *CAN DO WONDERS WITH MORE PARAMETER TUNING................. :-P
#---------------------------------------------------------------------------------------------

'''
