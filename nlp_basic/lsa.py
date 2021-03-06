import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer()

titles = [line.rstrip() for line in open('book_titles/all_book_titles.txt')]

# copy tokenizer from sentiment example
stopwords = set(w.rstrip() for w in open('book_titles/stopwords.txt'))
# add more stopwords specific to this problem
stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth', })

def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
    return tokens


word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []
error_count = 0
for title in titles:
	try:
		title = title.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
		all_titles.append(title)
		tokens = my_tokenizer(title)
		all_tokens.append(tokens)
		for token in tokens:
			if token not in word_index_map:
				word_index_map[token] = current_index
				current_index += 1
				index_word_map.append(token)
	except Exception as e:
		print(e)
		print(title)
		error_count += 1


print("Number of errors parsing file:", error_count, "number of lines in file:", len(titles))
if error_count == len(titles):
    print("There is no data to do anything with! Quitting...")
    exit()



#there are no lables so its unsupervised
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] = 1
    return x

N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N)) # terms will go along rows, documents along columns
i = 0
for tokens in all_tokens:
    X[:,i] = tokens_to_vector(tokens)
    i += 1

########### Linear    

# quite fast
#http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

# svd = TruncatedSVD()
# Z = svd.fit_transform(X)
# plt.scatter(Z[:,0], Z[:,1])
# for i in range(D):
# 	plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))
# plt.show()

#kinda slow. consists almost totaly 2 axis only few execptions (like in karesian cordinates with 0 in the center and +/- x/y all presnet)
#http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html

# from sklearn.decomposition import FactorAnalysis

# FA = FactorAnalysis()
# Z = FA.fit_transform(X)

# print("done")

# plt.scatter(Z[:,0], Z[:,1])
# for i in range(D):
# 	plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))
# plt.show()

# super slow does not converge in 200 iterations 
# kinda blured X in the middle
# independent component analisys
#http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

from sklearn.decomposition import FastICA, PCA

ICA = FastICA(max_iter=10000)
Z = ICA.fit_transform(X)
print("done")
plt.scatter(Z[:,0], Z[:,1])
for i in range(D):
	plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))
plt.show()

################ Non-Linear

#t-distributed Stochastic Neighbor Embedding t-SNE
#http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

# from sklearn.manifold import TSNE

# tSNE = TSNE(n_components=2)
# Z = tSNE.fit_transform(X)
# print("done")
# plt.scatter(Z[:,0], Z[:,1])
# for i in range(D):
# 	plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))
# plt.show()
