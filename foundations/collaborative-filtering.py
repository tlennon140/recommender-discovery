import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')

'''

Collaborative Filtering is based on the idea that users similar to a me can be used to 
predict how much I will like a particular product or service those users have 
used/experienced but I have not.

This implementation will use Surprise.

'''

reader = Reader()

ratings = pd.read_csv('data/ratings_small.csv')

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)

# Here we use the SVD implementation 
svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])

trainset = data.build_full_trainset()
svd.train(trainset)

ratings[ratings['userId'] == 1]

svd.predict(1, 302, 3)