from gensim.models.keyedvectors import KeyedVectors


if __name__ == '__main__':
    w2v =KeyedVectors.load_word2vec_format('glove500_w2v.txt', binary=False, unicode_errors='ignore')
    KeyedVectors.save_word2vec_format(w2v, 'glove500_w2v.bin', binary=True)