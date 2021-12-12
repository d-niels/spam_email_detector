import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model

DATA_DIRECTORY = 'spam-mail'
DATA_TYPE = 'spam'  # note: this spam file contains spam AND not spam


def create_txt(name, output):
    """
    create a text file and write to it
    :param name: str
    :param output: list[str]
    :return:
    """
    text_file = open(name, "w")
    for x in output:
        text_file.write(str(x.astype(int)[0]) + "\n")
    text_file.close()


class LanguageProcessor():
    """
    Keeps track of data and runs different machine learning algorithms on it,
    tests their effectiveness at predicting, and outputs the details to a text file.
    """
    def __init__(self, x_train, y_train, x_test, y_test):
        '''
        Pass
        :param x_train: Dataframe
        :param y_train: Dataframe
        :param x_test: Dataframe
        '''
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.vectorizer = None
        self.classifier = None

        self.uni = False
        self.bi = False
        self.uni_tfidf = False
        self.bi_tfidf = False

    def predict(self):
        """
        Predict using whichever algorithm has been set and saves results to a text file.
        :return: None
        """
        output = []
        score = 0
        score_0 = 0
        score_1 = 0
        total_0 = 0
        total_1 = 0

        for i in range(len(self.x_test)):
            output.append(self.classifier.predict(self.vectorizer.transform([self.x_test[i]])))
            if output[i] == self.y_test[i]:
                score += 1
            if self.y_test[i] == 0:
                total_0 += 1
                if output[i] == self.y_test[i]:
                    score_0 += 1
            if self.y_test[i] == 1:
                total_1 += 1
                if output[i] == self.y_test[i]:
                    score_1 += 1

        if self.uni:
            create_txt("output/" + DATA_DIRECTORY + "/unigram.output.txt", output)
            print("\nUNIGRAM:")
        if self.bi:
            create_txt("output/" + DATA_DIRECTORY + "/bigram.output.txt", output)
        if self.uni_tfidf:
            create_txt("output/" + DATA_DIRECTORY + "/unigramtfidf.output.txt", output)
        if self.bi_tfidf:
            create_txt("output/" + DATA_DIRECTORY + "/bigramtfidf.output.txt", output)

        print("overall accuracy:")
        print(str(score) + '/' + str(len(self.y_test)), "=", score/len(self.y_test))
        print("spam detection accuracy:")
        print(str(score_0) + '/' + str(total_0), "=", score_0/total_0)
        print("ham detection accuracy:")
        print(str(score_1) + '/' + str(total_1), "=", score_1/total_1)

    def train_uni_SGD(self):
        """
        Train a uni-gram SGD classifier.
        :return: None
        """
        print("\nTRAINING UNI SGD...")
        self.uni = True
        self.bi = False
        self.uni_tfidf = False
        self.bi_tfidf = False

        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        self.classifier = linear_model.SGDClassifier(loss="hinge", penalty="l1")
        self.classifier.fit(self.vectorizer.fit_transform(self.x_train), self.y_train)

    def train_bi_SGD(self):
        """
        Train a bi-gram SGD classifier.
        :return: None
        """
        print("\nTRAINING BI SGD...")
        self.uni = False
        self.bi = True
        self.uni_tfidf = False
        self.bi_tfidf = False

        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
        self.classifier = linear_model.SGDClassifier(loss="hinge", penalty="l1")
        self.classifier.fit(self.vectorizer.fit_transform(self.x_train), self.y_train)

    def train_uni_tfidf_SGD(self):
        """
        Train a uni-gram tfidf SGD classifier.
        :return: None
        """
        print("\nTRAINING TFIDF UNI SGD...")
        self.uni = False
        self.bi = False
        self.uni_tfidf = True
        self.bi_tfidf = False

        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
        self.classifier = linear_model.SGDClassifier(loss="hinge", penalty="l1")
        self.classifier.fit(self.vectorizer.fit_transform(self.x_train), self.y_train)

    def train_bi_tfidf_SGD(self):
        """
        Train a bi-gram tfidf SGD classifier.
        :return: None
        """
        print("\nTRAINING TFIDF BI SGD...")
        self.uni = False
        self.bi = False
        self.uni_tfidf = False
        self.bi_tfidf = True

        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        self.classifier = linear_model.SGDClassifier(loss="hinge", penalty="l1")
        self.classifier.fit(self.vectorizer.fit_transform(self.x_train), self.y_train)


if __name__ == "__main__":

    # get the training data
    train = pd.read_csv("data/" + DATA_DIRECTORY + "/" + DATA_TYPE + "_tr.csv")
    x_train = train["text"]
    y_train = train["polarity"]

    # get the testing data
    test = pd.read_csv("data/" + DATA_DIRECTORY + "/" + DATA_TYPE + "_te.csv")
    x_test = test["text"]
    y_test = test["polarity"]

    # create language processor
    lp = LanguageProcessor(x_train, y_train, x_test, y_test)

    # train and test on different classifiers
    lp.train_uni_SGD()
    lp.predict()

    lp.train_bi_SGD()
    lp.predict()

    lp.train_uni_tfidf_SGD()
    lp.predict()

    lp.train_bi_tfidf_SGD()
    lp.predict()

