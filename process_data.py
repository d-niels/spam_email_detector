import os
import re
import pandas as pd
import random as rand


def gather_files():
    """
    Creates a single dataframe containing whether or not the email is spam and its contents
    :return: pd.Dataframe, pd.Dataframe
    """
    tr_rows = []
    te_rows = []
    i = 0
    files = os.listdir("data/spam-mail/spam")
    cols = ['row_num', 'polarity', 'text']

    for file in files:
        f = open("data/spam-mail/spam/" + file, encoding ="ISO-8859-1")
        email = clean_text(f.read(), small_words=True)
        f.close()
        if rand.random() < .3:
            te_rows.append([i, 1, email])
        else:
            tr_rows.append([i, 1, email])
        i += 1
    print(i)

    files = os.listdir("data/spam-mail/ham")
    for file in files:
        f = open("data/spam-mail/ham/" + file, encoding ="ISO-8859-1")
        email = clean_text(f.read(), small_words=True)
        f.close()
        if rand.random() < .3:
            te_rows.append([i, 0, email])
        else:
            tr_rows.append([i, 0, email])
        i += 1
    print(i)

    return pd.DataFrame(te_rows, columns=cols), pd.DataFrame(tr_rows, columns=cols)


def clean_text(string, small_words=False):
    """
    Get the useless junk out of the text and make it all lowercase. All we want are words.
    :param string: str
    :param small_words: bool
    :return: list[list[str]]
    """
    stopwords = gather_stopwords()
    string = re.sub(" +", ",", re.sub("(\< ?(br|i|/i|em|spoiler|hr) ?\/?\>)|[^a-zA-Z ]", "", string)).lower()
    string = string.split(",")
    string = [x for x in string if x not in stopwords]
    output = ""
    for x in string:
        if small_words and len(x) > 25:
            pass
        else:
            output += x + " "
    return output


def gather_stopwords():
    """
    Read in the stopwords.
    :return: list[str]
    """
    f = open("stopwords.en.txt", "r")
    words = []
    for x in f:
        if x != "":
            words.append(re.sub("\s", "", x))
    return words


# get the reviews, clean them up, and save them to a csv
data_te, data_tr = gather_files()
data_te.to_csv('data/spam-mail/spam_te.csv', index=False)
data_tr.to_csv('data/spam-mail/spam_tr.csv', index=False)
