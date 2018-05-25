# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk import *
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from nltk.tokenize import RegexpTokenizer


class MovieReviews:

    def readText(filename):
        DF = pd.read_csv(filename, sep="\n", header=None)
        DF.columns = ['Reviews']
        return DF

    def splitData(pandasDataFrame):
        train, test = train_test_split(pandasDataFrame, test_size=0.20)
        return train, test

    def mergeData(pandasDataFrame1, pandasDataFrame2, pos, neg):
        DF1 = pandasDataFrame1.assign(label=pos)
        DF2 = pandasDataFrame2.assign(label=neg)
        df_new = pd.concat([DF1, DF2])
        return df_new

    def fitTrain(reviewTrainDF):
        count_vect = CountVectorizer()
        # learns reviews and labels
        X_train_counts = count_vect.fit_transform(reviewTrainDF['Reviews'])
        X_train_counts.shape
        # print('COUNT',X_train_counts)
        return X_train_counts

    def fitCount(Xtrain_count):
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(Xtrain_count)
        # print(X_train_tfidf)
        X_train_tfidf.shape

    def trainMultinomialNB(reviewTrainDF):
       # clf = MultinomialNB().fit(X_train_tfidf, reviewTrainDF['label'])
        stemmer = nltk.stem.SnowballStemmer('english')
        classified = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
        classified = classified.fit(reviewTrainDF.Reviews, reviewTrainDF.label)
        return classified

    def accuracyCal(text_clf):
        np.set_printoptions(threshold=np.nan)
        predicted = text_clf.predict(reviewTestDF.Reviews)
        realResult = np.array(reviewTestDF.label)
        print(predicted)
        print(realResult)
        result = np.mean((predicted) == np.array(reviewTestDF.label))
        return result

    def stem2(sentence):
        stemmer = nltk.stem.SnowballStemmer('english')
        words = nltk.word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalpha()]
        stems = [stemmer.stem(word) for word in words]
        words = " ".join(stems)
        # print('stems',words)
        return words

    def punc_rem(sentence):
        tokenizer = RegexpTokenizer(r'\w+')
        tokenized = tokenizer.tokenize(sentence)
        # tokenized = " ".join(tokenizer.tokenize(sentence))
        tokenized = [word.lower() for word in tokenized if word.isalpha()]
        tokenized = " ".join(tokenized)
        return tokenized


if __name__ == '__main__':
    movieObject = MovieReviews
    posReviewDF = movieObject.readText('/Users/ayberk/Desktop/posdata1')
    negReviewDF = movieObject.readText('/Users/ayberk/Desktop/negData1')
    posReviewDFTrain, posReviewDFTest = movieObject.splitData(posReviewDF)
    negReviewDFTrain, negReviewDFTest = movieObject.splitData(negReviewDF)
    reviewTrainDF = movieObject.mergeData(posReviewDFTrain, negReviewDFTrain, 1, 0)
    reviewTestDF = movieObject.mergeData(posReviewDFTest, negReviewDFTest, 1, 0)
    print(reviewTrainDF)
    # decreases the accuracy
    # reviewTrainDF['Reviews'] = reviewTrainDF['Reviews'].apply(movieObject.stem2)
    reviewTrainDF['Reviews'] = reviewTrainDF['Reviews'].apply(movieObject.punc_rem)
    # reviewTrainDF['Reviews'] = reviewTrainDF['Reviews'].apply(movieObject.listNgrams)
    fitTrainResult = movieObject.fitTrain(reviewTrainDF)
    movieObject.fitCount(fitTrainResult)
    classification_result = movieObject.trainMultinomialNB(reviewTrainDF)
    accuracy = movieObject.accuracyCal(classification_result)
    print('Bayes Accuracy: ',accuracy)
    print(reviewTrainDF)



