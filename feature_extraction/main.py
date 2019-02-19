from sklearn.naive_bayes import MultinomialNB
import json
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from bag_vectorizer import BagVectorizer


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
            x_vector.append(r['review'])
            y_vector.append(r['recommended'])
    return x_vector, y_vector


def train_model(x_train, y_train, bag_vectorizer):
    """ Train the model
    """
    # create a matrix with rows as texts and columns as tokens,
    # each cell containst the number of times the token appears in the text
    x_train_fit = bag_vectorizer.fit_transform()
    
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

def main():
    x_train, y_train = read_training_data('../data.json')

    bag_vectorizer = BagVectorizer(0, 0, 1, x_train) #TODO: might want to remove common words?
    classifier = train_model(x_train, y_train, bag_vectorizer)

    test_fit = bag_vectorizer.transform(['k', 'joke', 'good', 'bad', 'yaaaaaaaay'])
    print(classifier.predict_proba(test_fit)[:,1])

    words = get_words(bag_vectorizer)
    word_scores = get_scores(words, bag_vectorizer, classifier)

    words_tuple = create_tuples(words, word_scores)
    
    words_tuple.sort(key=lambda x: x[1])

    sorted_words = remove_scores(words_tuple)

    x_vector = []
    for review in x_train:
        review_vector = []
        review_tokens = bag_vectorizer.tokenize(review)
        for word in review_tokens:
            index = sorted_words.index(word)
            review_vector.append(index)
        x_vector.append(review_vector)

    x_vector.sort(key=len)
    median = len(x_vector[int(len(x_vector)/2)])

if __name__ == '__main__':
    main()