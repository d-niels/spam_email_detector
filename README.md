# imdb_review_predictor
Predicts whether a review is positive or negative by looking only at the text.

process_reviews.py is used to clean up the text in the reviews and turn it in to usable data. Punctuation is removed, all letters are made lowercase, and stopwords are removed.

The driver.py file is where all the learning happens. 50,000 movie review from imdb.com are used to train the classifiers. If a review was rated 7/10 or higher, it's considered a positive review. If a review was rated 4/10 or lower, its considered negative. Everything in between is considered neutral and isn't used for training. 
The classifiers used are stochastic gradient descent classifiers and its used in a unigram, bigram model, tf-idf unigram, and tf-idf bigram classifiers. The tf-idf is used to compensate for words that appear very frequently in all reviews (words that occur frequently in both positive and negative reviews).
The test data is then run through all the classifiers and predicted to be positive or negative. The results are output to text files.

imdb_tr.csv is where the processed reviews that are used for training are stored.

    1 - positive
    0 - negative

imdb_te.csv is the processed reviews that need to be predicted.

<classifier name>.output.txt is the file where the predictions are stored. Each line consists of a 1 or 0 to indicate whether the classifier predicted the review to be positive or negative. Once again, the numbers keep their same meaning:
    
    1 - positive
    0 - negative
