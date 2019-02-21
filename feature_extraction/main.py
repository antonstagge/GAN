from sklearn.naive_bayes import MultinomialNB
import json
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from bag_vectorizer import BagVectorizer
import numpy as np
import re


regex = r"[^!\"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~“”¨«»®´·º½¾¿¡§£₤‘’0-9A-Za-zöäå\s≈∞©€\\♥]"
pattern = re.compile(regex)
median = 33
padding = -1

def read_training_data(file):
    """ Convert file data to x and y vectors
    split each line into (first, text, score)
    x_vector - the vector of texts
    y_vector - the vector of scores
    """
    x_vector = []
    y_vector = []
    with open(file, 'r') as f:
        review_list = json.load(f)
        for r in review_list:
            rev = r['review']
            if(is_good(rev)):
                x_vector.append(r['review'])
                y_vector.append(r['recommended'])
    return x_vector, y_vector


def is_good(review):
    if pattern.search(review) is None:
        return True
    else:
        return False

def train_model(x_train, y_train, bag_vectorizer):
    """ Train the model
    """
    # create a matrix with rows as texts and columns as tokens,
    # each cell containst the number of times the token appears in the text
    x_train_fit = bag_vectorizer.fit_transform()
    bag_vectorizer.flag = False
    
    classifier = MultinomialNB()
    classifier.fit(x_train_fit, y_train)

    return classifier

def test_model(classifier, bag_vectorizer, x_test, y_test):
    """ Test the model accuracy by counting the 
    number of wrong predictions. Also prints the
    confusion matrix. 
    """
    x_test_fit = bag_vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test_fit)

    error = 0.0
    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            error += 1
    print("The model accuracy was: %.3f" % (1 - error/100))

    confusion = confusion_matrix(y_test, y_pred)
    print("pred-actual \n[[neg-neg pos-neg]\n[neg-pos pos-pos]]")
    print(confusion)

def get_words(bag_vectorizer):
    return bag_vectorizer.get_feature_names()
        
def get_scores(words, bag_vectorizer, classifier):
    words_fit = bag_vectorizer.transform(words)
    return classifier.predict_proba(words_fit)[:,1]

def create_tuples(words, word_scores):
    return list(zip(words, word_scores))

def remove_scores(words_tuple):
    words = []
    for t in words_tuple:
        words.append(t[0])
    return words

def normalize_review_length(x_vector, length):
    return_array = []
    for review_vector in x_vector:
        new_vector = []
        length_of_vector = len(review_vector)
        for i in range(length):
            if i < length_of_vector:
                new_vector.append(review_vector[i])
            else:
                new_vector.append(-1)
        return_array.append(new_vector)
    return return_array


def save_data_to_file(x_vector, filename):
    x_vector = np.array(x_vector)
    np.save(filename, x_vector)

def create_dict_from_list(list_of_words):
    dictionary = {}
    for i, word in enumerate(list_of_words):
        dictionary[word] = i
    return dictionary

def main():
    x_train, y_train = read_training_data('../data.json')

    print("Started bag")
    bag_vectorizer = BagVectorizer(0, 0, 1, x_train) #TODO: might want to remove common words?
    print("After bag")

    classifier = train_model(x_train, y_train, bag_vectorizer)
    print("After train CHO CHO")

    words = get_words(bag_vectorizer)
    word_scores = get_scores(words, bag_vectorizer, classifier)

    words_tuple = create_tuples(words, word_scores)
    
    words_tuple.sort(key=lambda x: x[1])

    sorted_words = remove_scores(words_tuple)
    save_data_to_file(sorted_words, '../dictionary')
    sorted_word_dict = create_dict_from_list(sorted_words)

    x_vector = []
    count = 0
    for review_tokens in bag_vectorizer.reviews_tokens:
        print("Count: %d" %(count))
        count +=1
        review_vector = []
        if len(review_tokens) == 0:
            continue
        for i, word in enumerate(review_tokens):
            if i == median:
                break
            index_in_sorted_list = sorted_word_dict[word]
            review_vector.append(index_in_sorted_list)
        if len(review_vector) < median:
            diff = median - len(review_vector)
            for i in range(diff):
                review_vector.append(padding)

        x_vector.append(review_vector)
    
    save_data_to_file(x_vector, '../featured_extracted_data')

if __name__ == '__main__':
    main()
