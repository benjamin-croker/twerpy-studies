import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


# load the tweets from file
tweets = pd.read_csv("tweets_sug.csv")

# merge all the tweets from a single user
grouped = tweets.groupby("screen_name")
print("Formatting data")
d = {"screen_name": [name for (name, data) in grouped],
     "tweets_text": [" ".join(list(data["text"])) for (name, data) in grouped],
     "tweet_group": [list(data["tweet_group"])[0] for (name, data) in grouped]
}

# remove all http links
# print("Removing links")
# d["tweets_text"] = [" ".join([w for w in tweets.lower().split(" ") if w[0:4] != "http"])
#                     for tweets in d["tweets_text"]]

# put the main work in a main loop, so the data can be accessed by importing the file
# if __name__ == "__main__":
# print("Removing stopwords")
# tokenizer = RegexpTokenizer(r'\w+')
# d["tweets_text"] = [" ".join([w for w in tokenizer.tokenize(tweets.lower())
#                               if w not in stopwords.words("english")])
#                   for tweets in d["tweets_text"]]
df = pd.DataFrame(d)

# filter out finance, since there's only 6 of them
df = df[df["tweet_group"] != "finance"]

# take a tfidf vectorisation of the text
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                    analyzer='word', token_pattern=r'\w{1,}',
                    decode_error = 'ignore',
                    ngram_range=(1, 1), use_idf=1, smooth_idf=1,
                    sublinear_tf=1)
lm = LogisticRegression()

print("Performing TFIDF transform")
X = tfv.fit_transform(df["tweets_text"])
y = np.array(list(df["tweet_group"]))

scores = cross_val_score(lm, X, y, cv=10, scoring="accuracy")
print("Mean Accuracy:{}, Std:{}".format(np.mean(scores), np.std(scores)))