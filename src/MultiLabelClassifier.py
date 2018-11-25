import scipy
import pandas as pd
import contractions, inflect
import datefinder
from scipy.io import arff
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_multilabel_classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class Lemmatizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, words):
        return [self.wnl.lemmatize(t) for t in words if len(t) > 1]

def preprocess(text):
    # NOTE: datefinder returns somewhat wrong indexes where the dates are located; it makes it unusable for our purpose
    # We want to replace datetime objects with a placeholder, but can't do that if the indexes are wrong
    matches = datefinder.find_dates(text, False, True)
    for match in matches:
        print(match)
        print(text)
        print(text[match[1][0]:match[1][1]])

    text = contractions.fix(text)
    words = word_tokenize(text)
    words = replace_numbers_with_placeholder(words)
    return words

def replace_numbers_with_string(words):
    #Replace all integer occurrences in list of tokenized words with its string representation
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def replace_numbers_with_placeholder(words):
    #Replace all integer occurrences in list of tokenized words with the placeholder <number>
    new_words = []
    for word in words:
        if word.isdigit():
            new_words.append('<number>')
        else:
            new_words.append(word)
    return new_words

df = pd.read_csv("Data/HCM.csv")
#df.head()

X = df["Utterance"]
y = df.drop("Utterance", axis=1)
#vect = CountVectorizer()
vect = TfidfVectorizer(preprocessor=preprocess, tokenizer=Lemmatizer())

# learn the vocabulary and transform it to a document-term-matrix
X_dtm = vect.fit_transform(X)

vect.get_feature_names()
# show all the features after they have been vectorized
pd.DataFrame(X_dtm.toarray(), columns=vect.get_feature_names())

# show all the labels
print(list(y))

#classifier = BinaryRelevance(MultinomialNB())
classifier = ClassifierChain(MultinomialNB())
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# train
classifier.fit(X_dtm, y)

userInput = input("Text to classify: ")
simple_test = [userInput]
simple_test_dtm = vect.transform(simple_test)

# predict
predictions = classifier.predict_proba(simple_test_dtm)
print(predictions)

#accuracy_score(y_test, predictions)
