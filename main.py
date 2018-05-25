# coding=utf-8
import nltk
import numpy as np
# nltk.download()
# nltk.download('all')
# nltk.download('stopwords')
# nltk.download('treebank')
# nltk.download('universal_tagset')
# nltk.download('averaged_perceptron_tagger')

from nltk import *
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer

def stem(sentence):
    stemmer = nltk.stem.SnowballStemmer('english')
    words = nltk.word_tokenize(sentence)
    words = [word.lower() for word in words if word.isalpha()]
    stems = [stemmer.stem(word) for word in words]
    print('stems',stems)
    return stems


def stopWord(words2):
    words=[]
    stopWords = set(stopwords.words('english'))
    for w in words2:
        if w not in stopWords:
            words.append(w)
    return words

def counter(sentence):
    most_freq = FreqDist(sentence)
    result = most_freq.most_common(10)
    return result

###PAHESE2#####

def listNgrams(cleanedReview,gram):
    ngramsResult = ngrams(cleanedReview, gram)
    # for gram in ngramsResult:
    #     print(gram)
    return ngramsResult

def listFreqBigram(words, freq, n):

    ignored_words = nltk.corpus.stopwords.words('english')
    bigram_measures = nltk.collocations.BigramAssocMeasures()

    finder = nltk.collocations.BigramCollocationFinder.from_words(words)
    finder.apply_freq_filter(freq)  # freq
    finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
    nbest_result = finder.nbest(bigram_measures.raw_freq, n)
    return nbest_result,finder,n


    # return nbest_result


def scoredBigram():
    nbest_result, finderX, n = listFreqBigram(clearedWord, 2, 3)
    bigram_measures = collocations.BigramAssocMeasures()
    #finder = BigramCollocationFinder.from_words(bigram)
    scored = finderX.score_ngrams(bigram_measures.raw_freq)
    # for i in scored:
    #     print(i[0])
    return scored


def sortedBigram():
    scoredList = scoredBigram()
    sorted_bigram_score = sorted(bigram for bigram, score in scoredList)
    return sorted_bigram_score

### PHASE 3

def pos_tagger(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(sentence)
    text = stopWord(tokenized)
    result = nltk.pos_tag(text)
    return result


def numOfTags(pos_tagged_list, a):
    # tag_fd = FreqDist(pos_tagged_list)
    # counts = tag_fd.most_common(a)
    tag_fd = nltk.FreqDist(tag for (word,tag) in pos_tagged_list)
    counts = tag_fd.most_common()
    return counts


def findWords(tagList, tag, num=15):

    common_tag_list = []
    if tag == 'noun':
        common_tag_list = [ tag_[0] for tag_ in tagList if tag_[1] == 'NN' or tag_[1] == 'NNS' or tag_[1] == 'NNP']
    elif tag == 'verb':
        common_tag_list = [ tag_[0] for tag_ in tagList if tag_[1] == 'VB' or tag_[1] == 'VBD' or tag_[1] == 'VBN']
    elif tag == 'adverb':
        common_tag_list = [tag_[0] for tag_ in tagList if tag_[1] == 'VB' or tag_[1] == 'VBD' or tag_[1] == 'VBN']
    elif tag == 'adjective':
        common_tag_list = [tag_[0] for tag_ in tagList if tag_[1] == 'JJR' or tag_[1] == 'JJ' or tag_[1] == 'JJS']
    elif tag == 'pronoun':
        common_tag_list = [tag_[0] for tag_ in tagList if tag_[1] == 'WP' or tag_[1] == 'PRP' or tag_[1] == 'PRP$']

    result = FreqDist(common_tag_list)
    result = result.most_common(num)
    result = numpy.array(result)
    return result[:,0]





list_of_all_reviews=[]
one_review=[]
votes =[]

def reader():
    readfile = open('rest.txt', "r")
    for line in readfile:
        Type = line.split('\t')
        x = Type[0]
        y = Type[1]
        print(" ")
        print('review ',y)
        list_of_all_reviews.append(y)
        votes.append(x)
        sentence = y
        print(" ")
        print(sentence)
        words = stem(sentence)
        print('stopwords', stopWord(words))
        global clearedWord
        clearedWord = stopWord(words)
        print('top 10 ', counter(clearedWord))
        nGram = listNgrams(clearedWord, 2)
        print('N gram ', list(nGram))
        print('N best Bigram', list(listFreqBigram(clearedWord, 2, 3))[0])
        # nbest_bigram = listFreqBigram(clearedWord,2,5)
        scored_bigram = scoredBigram()
        print('scored bigram', scored_bigram)
        # scored_result = scoredBigram(words)
        print('sorted bigram', sortedBigram())
        POS = pos_tagger(sentence)
        print('POS', POS)
        print("Num of POS", numOfTags(POS, 10))
        print('FindWord',findWords(POS,'noun', 5))

reader()


