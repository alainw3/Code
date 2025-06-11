import gensim.downloader as api

from gensim.models import Word2Vec
import numpy as np

#import gensim.models.word2vec as w2v

corpus = api.load("word2vec-google-news-300") #download the model and return as object ready to use

#model = Word2Vec(corpus)
#model.build_vocab(corpus)

#model = Word2Vec.load("word2vec-google-news-300")
word_vectors = corpus
#word_vectors = model


# Let^'s look how the vector embedding looks like
#print(word_vectors['computer']) # Eaxmple : computer
 
word1  = 'man'
word2  ='woman'

vector_difference1 = word_vectors[word1] -  word_vectors[word2]

magnitude_of_difference = np.linalg.norm(vector_difference1)

print (magnitude_of_difference)


#print(word_vectors.wv.most_similar(positive=['king','woman'], negative=['man']))




