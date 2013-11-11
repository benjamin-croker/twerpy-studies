import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

SEED = 42


def remove_http_links(concat_tweets):
    print("Removing HTTP links")
    return [" ".join([w for w in tweets.lower().split(" ") if w[0:4] != "http"])
            for tweets in concat_tweets]


def remove_stopwords(concat_tweets):
    print("Removing Stopwords")
    tokenizer = RegexpTokenizer(r'\w+')
    return [" ".join([w for w in tokenizer.tokenize(tweets.lower())
                      if w not in stopwords.words("english")])
            for tweets in concat_tweets]


def load_raw_tweets(load_csv="tweets_sug.csv", write_csv="tweets_processed.csv",
                    rm_links=True, rm_stopwords=True):
    # load the tweets from file
    tweets = pd.read_csv(load_csv)

    # merge all the tweets from a single user
    grouped = tweets.groupby("screen_name")
    print("Formatting data")
    d = {"screen_name": [name for (name, data) in grouped],
         "tweets_text": [" ".join(list(data["text"])) for (name, data) in grouped],
         "tweet_group": [list(data["tweet_group"])[0] for (name, data) in grouped]}

    if rm_links:
        d["tweets_text"] = remove_http_links(d["tweets_text"])

    if rm_stopwords:
        d["tweets_text"] = remove_stopwords(d["tweets_text"])

    df = pd.DataFrame(d)

    # filter out finance, since there's only 6 of them
    df = df[df["tweet_group"] != "finance"]
    df.to_csv(write_csv, index=False)
    return df


def load_processed_tweets(load_csv="tweets_processed.csv"):
    return pd.read_csv(load_csv)


def eval_model(df):
    # perform k-fold validation
    kf = KFold(n=df.shape[0], n_folds=10, random_state=SEED, shuffle=True)
    acc_scores_log = np.zeros(10)
    acc_scores_rf = np.zeros(10)
    acc_scores_comb = np.zeros(10)

    fold_n = 0
    
    # logistic regression model with defaults
    log_cl = LogisticRegression()
    # rf model
    rf_cl = RandomForestClassifier(n_estimators=100, min_samples_split=16, random_state=SEED)
    
    for train_indices, fold_eval_indices in kf:
        print("Evaluating fold {} of {}".format(fold_n+1, 10))
        # take a tfidf vectorisation of the text
        tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                              analyzer='word', token_pattern=r'\w{1,}',
                              decode_error='ignore',
                              ngram_range=(1, 1), use_idf=1, smooth_idf=1,
                              sublinear_tf=1)

        X_train = tfv.fit_transform(df["tweets_text"][train_indices])
        X_eval = tfv.transform(df["tweets_text"][fold_eval_indices])

        y_train = np.array(list(df["tweet_group"][train_indices]))
        y_eval = np.array(list(df["tweet_group"][fold_eval_indices]))

        log_cl.fit(X_train, y_train)
        log_preds = log_cl.predict(X_eval)
        log_proba = log_cl.predict_proba(X_eval)
        acc_scores_log[fold_n] = accuracy_score(y_eval, log_preds)

        # use the most important words to train RF classifier
        # take the max absolute value from all one-v-all subclassifiers
        coef = np.abs(log_cl.coef_).mean(0)
        important_words_ind = np.argsort(coef)[-200:]

        X_train_dense = X_train[:, important_words_ind].todense()
        X_eval_dense = X_eval[:, important_words_ind].todense()

        rf_cl.fit(X_train_dense, y_train)
        rf_preds = rf_cl.predict(X_eval_dense)
        rf_proba = rf_cl.predict_proba(X_eval_dense)
        acc_scores_rf[fold_n] = accuracy_score(y_eval, rf_preds)

        # combine predictions by taking the maximum probabilities from both classifiers
        if not all(log_cl.classes_ == rf_cl.classes_):
            print("Error: different classes for classifiers. Combined predictions incorrect")
        comb_proba = np.maximum(log_proba, rf_proba)
        comb_preds = [log_cl.classes_[i] for i in comb_proba.argmax(1)]
        acc_scores_comb[fold_n] = accuracy_score(y_eval, comb_preds)

        fold_n += 1

    print("Mean Log Accuracy:{}, Std:{}".format(np.mean(acc_scores_log), np.std(acc_scores_log)))
    print("Mean RF Accuracy:{}, Std:{}".format(np.mean(acc_scores_rf), np.std(acc_scores_rf)))
    print("Mean Combined Accuracy:{}, Std:{}".format(np.mean(acc_scores_comb), np.std(acc_scores_comb)))

if __name__ == "__main__":
    #df = load_raw_tweets()
    df = load_processed_tweets()
    eval_model(df)
