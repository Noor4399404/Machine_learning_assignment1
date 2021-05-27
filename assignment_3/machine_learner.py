import argparse
import pprint
from nltk import word_tokenize
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score


def decision_tree(documents, labels, ratio):

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(documents, labels, test_size=ratio, random_state=0)

    clf = Pipeline([('vec', TfidfVectorizer()), ('cls', DecisionTreeClassifier())])

    clf.fit(Xtrain, Ytrain)

    Yguess = clf.predict(Xtest)

    print('This is the classification report:\n')
    print(classification_report(Ytest, Yguess, digits=3, zero_division=1), "\n\n")
    print("This is the confusion matrix:\n")
    print(confusion_matrix(Ytest, Yguess))


def naive_bayes(documents, labels, ratio):

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(documents, labels, test_size=ratio, random_state=0)

    clf = Pipeline([('vec', TfidfVectorizer()), ('cls', MultinomialNB())])

    clf.fit(Xtrain, Ytrain)

    Yguess = clf.predict(Xtest)

    print('This is the classification report:\n')
    print(classification_report(Ytest, Yguess, digits=3, zero_division=1), "\n\n")
    print("This is the confusion matrix:\n")
    print(confusion_matrix(Ytest, Yguess))


def read_corpus(filepath):

    documents = []
    labels = []

    with open(filepath, 'r', encoding='utf-8') as inp:
        text = inp.readlines()
    
    for line in text:
        tokens = word_tokenize(line)
        labels.append(tokens[0])
        documents.append(' '.join(tokens[2:]))

    return documents, labels


def main():

    parser = argparse.ArgumentParser(
        description="Can be used to test different ways to learn a machine emotion detection.")
    parser.add_argument(
        "filepath",
        help="The path to the trainset .txt file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--tree', action='store_true', help='This will estimate the emotions by making a decision tree.')
    group.add_argument('-b', '--bayes', action='store_true', help='This will estimate the emotions using Naive Bayes.')
    group.add_argument('-c', '--cross_validation', default=argparse.SUPPRESS, help='Prints different cross validation scores. Takes a number that detemines the splitting strategy.')

    args = parser.parse_args()

    documents, labels = read_corpus(args.filepath)

    if args.tree:
        decision_tree(documents, labels, 0.25)

    if args.bayes:
        naive_bayes(documents, labels, 0.25)

    if args.cross_validation:

        strategy = int(args.cross_validation)
        
        clf = Pipeline([('vec', TfidfVectorizer()), ('cls', DecisionTreeClassifier())])
        scores = cross_val_score(clf, documents, labels, cv=strategy)
        print(f'The decision tree algorithm has a {scores.mean():.2f} accuracy with a standard deviation of {scores.std():.2f}')

        clf = Pipeline([('vec', TfidfVectorizer()), ('cls', MultinomialNB())])
        scores = cross_val_score(clf, documents, labels, cv=strategy)
        print(f'The decision tree algorithm has a {scores.mean():.2f} accuracy with a standard deviation of {scores.std():.2f}')

    
if __name__ == "__main__":
    main()
