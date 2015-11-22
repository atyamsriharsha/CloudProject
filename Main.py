import sys
from pprint import pprint
import os
from pyspark import SparkContext
from operator import add
import pandas as pd
from pyspark.mllib.feature import HashingTF, IDF, Vectors
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LabeledPoint
from nltk.corpus import stopwords
import nltk
import yaml
import csv
from random import shuffle

class Person:
	"""docstring for Person"""
	def parse(self,line):
		fields =line.split('\t')
		self.name=fields[0]
		self.country=fields[1]
		self.gender =fields[2]
		self.tag = fields[3]
		self.post_name =fields[4] 
		self.comment = fields[5]
		
sc = SparkContext(appName="Dividing")
NumberOfPostsForEachPost = sc.textFile(sys.argv[1],1).map(lambda x:x.split('\t'))
NumberOfPostsForEachPost=NumberOfPostsForEachPost.map(lambda t:(t[4],1)).reduceByKey(lambda x,y:x+y)
NumberofTagsForEachTag = sc.textFile(sys.argv[1],1).map(lambda x:x.split('\t')).map(lambda t:(t[3],1)).reduceByKey(lambda x,y:x+y)
NumberofTagsForEachPost = sc.textFile(sys.argv[1],1).map(lambda x:x.split('\t')).map(lambda t:(t[4],str(t[3]))).reduceByKey(lambda x,y:str(x)+'\n'+str(y))
NumberofCommentsForEachPost = sc.textFile(sys.argv[1],1).map(lambda x:x.split('\t')).map(lambda t:(t[4],str(t[5]))).reduceByKey(lambda x,y:str(x)+'\n'+str(y))
NumberofCommentsForEachTag = sc.textFile(sys.argv[1],1).map(lambda x:x.split('\t')).map(lambda t:(t[3],str(t[5]))).reduceByKey(lambda x,y:str(x)+'\n'+str(y))

list1 = NumberofTagsForEachTag.collect()
list2 = NumberofCommentsForEachTag.collect()


TopTopic = 0
HighestcountedTag = NumberofTagsForEachTag.map(lambda x:x[1]).max()
for x in xrange(0,len(list1)):
	if list1[x][1]==HighestcountedTag:
		TopTopic = list1[x][0]
		print "This is the Top trending Topic ",list1[x][0][1:]," and the Number of tags are", HighestcountedTag,"\n"
		break
print "This is Minimum number of tags for the post \n",NumberofTagsForEachTag.map(lambda x:x[1]).min()

commentlist = 0
for x in xrange(0,len(list2)):
	if list2[x][0]==str(TopTopic):
		commentlist = str(list2[x][1])


with open('data.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    data = commentlist.split('\n')
    a.writerows(data)
text = str(commentlist.split('\n'))

class Splitter(object):

    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences


class POSTagger(object):

    def __init__(self):
        pass
    def pos_tag(self, sentences):
        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        #adapt format
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos

class DictionaryTagger(object):

    def __init__(self, dictionary_paths):
        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        map(lambda x: x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))

    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):
        """
        the result is only one tagging of all the possible ones.
        The resulting tagging is determined by these two priority rules:
            - longest matches have higher priority
            - search is made from left to right
        """
        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N) #avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                if tag_with_lemmas:
                    literal = expression_lemma
                else:
                    literal = expression_form
                if literal in self.dictionary:
                    #self.logger.debug("found: %s" % literal)
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence

def value_of(sentiment):
    if sentiment == 'positive': 
    	return 1
    if sentiment == 'negative': 
    	return -1
    return 0

def sentence_score(sentence_tokens, previous_token, acum_score):    
    if not sentence_tokens:
        return acum_score
    else:
        current_token = sentence_tokens[0]
        tags = current_token[2]
        token_score = sum([value_of(tag) for tag in tags])
        if previous_token is not None:
            previous_tags = previous_token[2]
            if 'inc' in previous_tags:
                token_score *= 2.0
            elif 'dec' in previous_tags:
                token_score /= 2.0
            elif 'inv' in previous_tags:
                token_score *= -1.0
        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)

def sentiment_score(review):
    return sum([sentence_score(sentence, None, 0.0) for sentence in review])

if __name__ == "__main__":
    splitter = Splitter()
    postagger = POSTagger()
    dicttagger = DictionaryTagger([ 'dicts/positive.yml', 'dicts/negative.yml', 
                                    'dicts/inc.yml', 'dicts/dec.yml', 'dicts/inv.yml'])
    splitted_sentences = splitter.split(text)
    #pprint(splitted_sentences)
    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
    #pprint(pos_tagged_sentences)
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    #pprint(dict_tagged_sentences)

    print("analyzing sentiment...")
    score = sentiment_score(dict_tagged_sentences)
    print(score)
    if score>0:
    	print "Our post has Positive reviews"
    elif score<0:
    	print "Bad reviews :("
    else:
    	print "Neutral State"
