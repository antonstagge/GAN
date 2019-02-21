import time
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer

# bag_size = 1
# number_of_common_words = 10
# common_words = []
punctuation_string = r"[!\"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~“”¨«»®´·º½¾¿¡§£₤‘’0-9]"

class BagVectorizer:
    def __init__(self, n_c_w, n_l_c_w, bag_size, x_train):
        self.n_common_words = n_c_w
        self.n_least_common_words = n_l_c_w
        self.bag_size = bag_size
        self.x_train = x_train
        self.words_to_remove = []
        self.reviews_tokens = []
        self.flag = True
        
        # self.set_words_to_remove()
        self.vectorizer = CountVectorizer(tokenizer=self.simple_tokenizer)

    def fit_transform(self):
        return self.vectorizer.fit_transform(self.x_train)
    
    def transform(self, x_vector):
        return self.vectorizer.transform(x_vector)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


    def set_words_to_remove(self):
        """ Count the number of times each word appears and then
        place the top number_of_common_words in the 
        global variable common_words
        """
        vectorizer = CountVectorizer(tokenizer=self.simple_tokenizer)
        x_train_fit = vectorizer.fit_transform(self.x_train)
        
        words= vectorizer.get_feature_names()

        word_count = []
        for i in range(x_train_fit.shape[1]):
            count = np.sum(x_train_fit.getcol(i))
            tup = (count, words[i])
            word_count.append(tup)

        def comparator(tupEl):
            # Sort on the count
            return tupEl[0]

        
        word_count.sort(key=comparator)

        # Add most common words
        for i in range(1, self.n_common_words):
            self.words_to_remove.append(word_count[-i][1])

        # Add least common words
        for i in range(1, self.n_least_common_words):
            self.words_to_remove.append(word_count[i][1])



    def simple_tokenizer(self, text):
        """ Remove any type of punctuation and the words 
        and then split on whitespace
        """
        
        re_tok = re.compile(punctuation_string)
        tokens = re_tok.sub(' ', text).split()
        if self.flag:
            self.reviews_tokens.append(tokens)
        return tokens


    def tokenize(self, text):
        """ Remove any type of punctuation and the words 
        contained in the global variable common_words
        and then split on whitespace
        """
        common_words_string = " | ".join(self.words_to_remove)
        re_tok = re.compile(punctuation_string + "| " + common_words_string + " ")
        words = re_tok.sub(' ',re_tok.sub(' ',text)).split()

        tokens = []
        for i in range(len(words)-(self.bag_size-1)):
            bag = []
            for j in range(self.bag_size):
                bag.append(words[i + j].lower())
            tokens.append(' '.join(bag))
        return tokens

if __name__ == '__main__':
    # For testing
    test = "Before I begin I'd just like point out that I am not reviewing this film as a work of \"\"art\"\" -- on that score, it seems just as good as most films, if not at least a little better -- but as a work of propaganda."
    test = test.lower()
    print(test)
    print(tokenize(test))
