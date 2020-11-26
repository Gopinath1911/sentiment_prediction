from textblob import TextBlob 
import re 
import pandas as pd
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


df=pd.read_csv("airline_sentiment_analysis.csv",encoding='latin1')
df=df.loc[:,"airline_sentiment":"text"]


df['text'] = df["text"].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

#nltk.download('stopwords')
# Global Parameters
stop_words = set(stopwords.words('english'))

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

tf_vector = get_feature_vector(np.array(df.iloc[:, 1]).ravel())
X = tf_vector.transform(np.array(df.iloc[:, 1]).ravel())
y = np.array(df.iloc[:, 0]).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)


# Training Logistics Regression model
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train, y_train)
y_predict_lr = LR_model.predict(X_test)

def sentiment_prediction(text): 
   X_test=[text]
   print(X_test)
   test_df=pd.DataFrame()
   test_df["sentiment"]=["unknown"]
   test_df["test"]=X_test

   test_feature = tf_vector.transform(np.array(test_df.iloc[:, 1]).ravel())

   # Using Logistic Regression model for prediction
   test_prediction_lr = LR_model.predict(test_feature)

   # Averaging out the hashtags result
   test_result_ds = pd.DataFrame({'hashtag': "#", 'prediction':test_prediction_lr})
   test_result = test_result_ds.groupby(['hashtag']).max().reset_index()
   test_result.columns = ['heashtag', 'predictions'] 
   res=test_result.predictions
   res = res.to_string(index = False)
   return str(res)




