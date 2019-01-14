# https://www.kaggle.com/snap/amazon-fine-food-reviews
import warnings
#ignore the unnesessory warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import nltk     #natural language processing tool-kit
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer                 # Stemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
import re
# from gensim.models import Word2Vec               s                    #For Word2Ve

class ReviewClassifier():
    """docstring for ReviewClassifier."""
    def __init__(self):    
        review_path = "input/Reviews.csv"
        #read the csv file
        data = pd.read_csv(review_path)
        self.final = self.sterilize_data(data)
        # self.X = self.final['Text']
        self.stop = self.words()
        self.stem_input()
        # self.final['Text'] = self.X
        self.final = self.final[['Text','Score']]
        X = self.final.Text
        y = self.final.Score
        self.TFIDF()
        #set the training and testing data
        self.X_Train,  self.X_Test, self.y_train, self.y_test = train_test_split(X, y, random_state = 0)
        self.LSTM_neural_network(self.final)
    def LSTM_neural_network(self, data_frame):
        max_features = 30000
        tokenizer = Tokenizer(nb_words=max_features, split=' ')
        tokenizer.fit_on_texts(data_frame['Text'].values)
        X1 = tokenizer.texts_to_sequences(data_frame['Text'].values)
        X1 = pad_sequences(X1)
        Y1 = pd.get_dummies(data_frame['Score'].values)
    def TFIDF(self):
        tfidf = TfidfVectorizer()
        tfidf.fit(self.final['Text'])
        X = tfidf.transform(self.final['Text'])
    def sterilize_data(self, df):
        #only use the first 5000 rows
        data = df.head(5000)
        #we want to seperate it into positive and negative scores, so we seperate it from 1-2 and 4-5
        #drop any score thats 3
        data = self.remove_score(data,3)
        #get all the scores
        scores = data['Score']
        #map the words positive and negative to the array n, this will take the scores and give it a positive or negative value based on if its higher or lower than 3
        n = scores.map(self.seperate_scores)
        #add n to the scores col in the data frame
        data['Score'] = n
        return self.remove_duplicates(data)


    def remove_duplicates(self,df):
        #remove any duplicate data in the dataframe, if the same userID posted a text at the same time drop that row as this is clearly a duplicate entery
        data = df.drop_duplicates(subset={"UserId",'ProfileName',"Time","Text"})
        # HelpfulnessNumerator should always be less than or equal to HelpfulnessDenominator so checking this condition and removing those records also.
        return data[data['HelpfulnessNumerator'] <= data['HelpfulnessDenominator']]

    def remove_score(self, df, n):
        #remove the score n from the dataframe
        return df[df['Score'] != n]
    
    def seperate_scores(self,x):
        #classify it positive or negative 
        return 1 if x > 3 else 0
    def stem_input(self):
        temp = []
        snow = nltk.stem.SnowballStemmer('english')
        for n in self.final['Text']:
            n = n.lower()
            cleaner = re.compile('<.*?>')   #remove any html links from the words
            n = re.sub(cleaner, ' ',n)
            n = re.sub(r'[?|!|\'|"|#]',r'',n) 
            n = re.sub(r'[.|,|)|(|\|/]',r' ',n)     #remove any punctuations
            words = [snow.stem(word) for word in n.split() if word not in self.stop]    #stem and remove stop words
            temp.append(words)
        sent = []
        for row in temp:
            sentence = ''
            for word in row:
                sentence = sentence + " " + word
            sent.append(sentence)
        self.final['Text'] = sent
    def words(self):
        return {'in', 'below', "aren't", 'them', 'be', 'needn', 'as', 'into', 'is', 'haven', 'o', "hadn't", 'few', 'until', 'she', 'for', 'his', 'do', 'what', 'again', 'mustn', "that'll", 'yourself', 're', 'most', 'y', "haven't", 'where', 'own', 'about', 'yourselves', 'before', "hasn't", "mustn't", 'an', 'been', "she's", 'hers', 'which', 'was', 'did', 'with', 'from', 'themselves', 'ourselves', 'we', "shouldn't", 'doing', 'should', 'between', 't', 'further', 'wasn', 'him', 'not', 'those', 'other', 'doesn', "weren't", 'your', 'don', 'my', 'that', 'd', 'you', 'there', 'any', 'very', 'only', 'who', 'through', 'i', 'up', 'same', 'after', 'the', 'why', 'ours', 'out', 'theirs', 'to', 'hadn', 'couldn', 'at', 'her', 'some', 'have', 'here', 'our', 'myself', 'once', "wouldn't", "you'll", "don't", 's', 'how', 'm', 'by', 'such', 'will', 'each', 'while', 'me', "doesn't", 'when', "you'd", 'these', 'it', 'no', "wasn't", 'just', 'than', 'or', 'having', 'itself', 'too', 'now', 'on', 'himself', 'won', 'down', 'so', "couldn't", 'but', 'hasn', "didn't", "it's", 'its', "isn't", 'weren', 'whom', 'shan', "won't", 'and', 'being', 'herself', 'ma', 'over', 'll', 'are', 've', 'off', 'has', 'ain', 'aren', 'both', 'nor', 'didn', 'does', "shan't", 'he', 'against', 'then', 'yours', 'all', 'during', 'under', 'mightn', 'isn', 'this', 'wouldn', 'above', "mightn't", 'their', 'am', "should've", 'a', 'more', 'of', 'had', 'were', 'because', "you've", 'they', "needn't", 'shouldn', "you're", 'if', 'can'}
ReviewClassifier()