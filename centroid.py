from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
# from commons import get_max_length, similarity, average_score, clean, get_wc
import os
import itertools

stop = set(stopwords.words('english'))   # Because of how nltk's word_tokenize works.
stop.add("''")
stop.add("``")
stop.add("'s")


class Summarizer(object):
    def __init__(self,
                 w2v_path= os.path.dirname(os.path.realpath(__file__)) + '/glove/glove500_w2v.bin',
                 length_limit=10,
                 threshold=0.3,
                 sim_threshold=0.95):
        self.length_limit = length_limit
        self.stop = stop
        self.threshold = threshold
        self.sim_threshold = sim_threshold
        print("****** LOADING W2V ******")
        self.w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True, unicode_errors='ignore')
        self.index2words = set(self.w2v.wv.index2word)
        print("****** FINISHED LOADING W2V ******")
        self.word_vectors = {}

    def not_stop(self, word):
        return word not in self.stop and word not in string.punctuation

    def sent_tokenize(self, text):
        article = sent_tokenize(text)
        article = [clean(t) for t in article]
        return [t for t in article if len(t) > self.length_limit and t[-1] != ':']

    def preprocess_text(self, sentences):
        cleaned_sentences = []
        for sentence in sentences:
            words = [t.lower() for t in word_tokenize(sentence) if self.not_stop(t)]
            cleaned_sentences.append(' '.join(words))
        return cleaned_sentences

    def get_centroid_words(self, sentences):
        tfidf_vect = TfidfVectorizer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = tfidf_vect.fit_transform(sentences)
        tfidf = tfidf.toarray()
        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())     # Normalize values, by doing this
        vocab = tfidf_vect.get_feature_names()                                  # the threshold works.
        c_word_list = []
        for x in range(centroid_vector.shape[0]):
            if centroid_vector[x] > self.threshold:
                c_word_list.append(vocab[x])
            else:
                centroid_vector[x] = 0

        return tfidf, c_word_list, centroid_vector

    def init_word_vector(self, sentences):
        self.word_vectors = {}
        for sen in sentences:
            for word in sen.split():
                if word in self.index2words:
                    self.word_vectors[word] = self.w2v[word]
        return

    def get_word_vector(self, words):
        vectors = np.zeros(self.w2v.vector_size, dtype='float32')
        vector_keys = set(self.word_vectors.keys())
        count = 0
        for word in words:
            if word in vector_keys:
                vectors = vectors + self.word_vectors[word]
                count += 1
        if count != 0:
            vectors = np.divide(vectors, count)     # Normalize
        return vectors

    def get_sentence_scores_multiple(self, sentences, raw_sentences, centroid_vector):
        length_param = 0.15
        sentence_scores = []
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                scores = []
                words = sentences[i][j].split()
                sen_vector = self.get_word_vector(words)
                scores.append(similarity(sen_vector, centroid_vector))
                scores.append(length_param * (1 - (j / len(sentences[i]))))

                score = average_score(scores)
                sentence_scores.append(((i, j), raw_sentences[i][j], score, sen_vector))

        return sorted(sentence_scores, key=lambda elem: elem[2], reverse=True)

    def get_sentence_scores(self, sentences, raw_sentences, centroid_vector):
        sentence_scores = []
        for i in range(len(sentences)):
            scores = []
            words = sentences[i].split()
            sen_vector = self.get_word_vector(words)
            scores.append(similarity(sen_vector, centroid_vector))
            # scores.append()
            # Here we can append different scores and change scoring of sentences.. Such as BoW-param, length & position
            score = average_score(scores)
            sentence_scores.append((i, raw_sentences[i], score, sen_vector))    # index, rå, score, centroid_sentence
            # TODO get similarity to the centroid vector
        return sorted(sentence_scores, key=lambda elem: elem[2], reverse=True)

    def scoring_rossellio(self, sorted_sentence_scores, count, limit, limit_type, word_count):
        sentence_summary = []
        for sen_item in sorted_sentence_scores:     # TODO here we could let ourself
            if count > limit:                       # retreive first sentence if that is important.
                break
            include = True
            for summary_item in sentence_summary:   # Use of too similar vectors
                sim_score = similarity(sen_item[3], summary_item[3])
                if sim_score > self.sim_threshold:
                    include = False
            if include:
                sentence_summary.append(sen_item)
                if limit_type == 'word':
                    count += len(sen_item[1].split())
                else:
                    count += len(sen_item[1].split())/word_count
        return sentence_summary

    def scoring_ghalandari(self, count, limit, limit_type, word_count, centroid, score_sentences):
        sentence_summary = [score_sentences[0]]
        chosen_sen = set([score_sentences[0][0]])      # Index can be input here
        centroid_summary = score_sentences[0][3]
        """
        1) Addera sentence_centroid med summary_centroid (tmp) --> jmf med centroid, ta ut score
        2) If max & include & !inChosenSen --> max_index = index
        3) 
        """

        while count < limit:
            sentence_quadruple = (None, None, None, None)
            max_score = 0
            for sen_item in score_sentences:    # (index, raw, score, sentence_centroid)
                include = True
                for summary_item in sentence_summary:   # Use of too similar vectors
                    sim_score = similarity(sen_item[3], summary_item[3])
                    if sim_score > self.sim_threshold:
                        include = False
                tmp_summary_centroid = sen_item[3] + centroid_summary
                tmp_summary_centroid = np.divide(tmp_summary_centroid, 2)
                score = similarity(tmp_summary_centroid, centroid)
                if score > max_score and include and sen_item[0] not in chosen_sen:
                    sentence_quadruple = sen_item
                    max_score = score

            if sentence_quadruple[0] is None:
                return sentence_summary

            sentence_summary.append(sentence_quadruple)
            chosen_sen.add(sentence_quadruple[0])
            if limit_type == 'word':
                count += len(sentence_quadruple[1].split())
            else:
                count += len(sentence_quadruple[1].split()) / word_count
        return sentence_summary

    def sent_rework(self, sent_list):
        reworked_sentences = []
        for sen in sent_list:
            sen_split = sen.split(',', maxsplit=1)
            if len(sen_split) > 1 and len(sen_split[0]) > len(sen_split[1]):
                reworked_sen = sen_split[0]
            else:
                reworked_sen = sen
            reworked_sentences.append(reworked_sen)
        print(sum([len(x) for x in sent_list]))
        print(sum([len(x) for x in reworked_sentences]))
        return reworked_sentences

    def summarize(self, text, limit_type='word', limit=100.0):        # Limit type = word and percent
        print("****** PREPROCESSING ******")
        word_count = len(text.split())
        raw_sentence_list = self.sent_tokenize(text)
        #raw_sentence_list = self.sent_rework(raw_sentence_list) # TODO how about removing transition phrase here?
        clean_sentences = self.preprocess_text(raw_sentence_list)

        print("****** CREATING CENTROID ******")
        tfidf, centroid_words, centroid_bow = self.get_centroid_words(clean_sentences)
        self.init_word_vector(clean_sentences)
        centroid_vector = self.get_word_vector(centroid_words)

        print("****** SENTENCE SCORING ******")

        sorted_sentence_scores = self.get_sentence_scores(clean_sentences, raw_sentence_list, centroid_vector)

        print("****** SENTENCE SELECTION ******")
        count = 0

        # sentence_summary = self.scoring_rossellio(sorted_sentence_scores, count, limit, limit_type, word_count)#[]
        sentence_summary = self.scoring_ghalandari(count, limit, limit_type, word_count, centroid_vector, sorted_sentence_scores)

        #sentence_summary = sorted(sentence_summary, key=lambda elem: elem[0], reverse=False)    # Chronological ordering
        # TODO make sorted a attribute of class, if single article we want sorted otherwise probably not.
        summary = "\n".join([s[1] for s in sentence_summary])

        return summary

    def summarize_multiple(self, articles, limit_type='word', limit=100.0):        # Limit type = word and percent
        print("****** PREPROCESSING ******")
        word_count = sum([len(article.split()) for article in articles])

        raw_sentence_list = [self.sent_tokenize(article) for article in articles]
        #raw_sentence_list = self.sent_rework(raw_sentence_list) # TODO how about removing transition phrase here?
        clean_sentences = [self.preprocess_text(raw_sentences) for raw_sentences in raw_sentence_list]

        print("****** CREATING CENTROID ******")
        merged_clean = list(itertools.chain.from_iterable(clean_sentences))
        tfidf, centroid_words, centroid_bow = self.get_centroid_words(merged_clean)
        self.init_word_vector(merged_clean)
        centroid_vector = self.get_word_vector(centroid_words)

        print("****** SENTENCE SCORING ******")

        sorted_sentence_scores = self.get_sentence_scores_multiple(clean_sentences, raw_sentence_list, centroid_vector)

        print("****** SENTENCE SELECTION ******")
        count = 0

        sentence_summary = self.scoring_rossellio(sorted_sentence_scores, count, limit, limit_type, word_count)#[]

        #sentence_summary = sorted(sentence_summary, key=lambda elem: elem[0], reverse=False)    # Chronological ordering
        # TODO make sorted a attribute of class, if single article we want sorted otherwise probably not.
        summary = "\n".join([s[1] for s in sentence_summary])

        return summary


def get_max_length(sentences):
    sen_lengths = [len(s.split()) for s in sentences]
    return max(sen_lengths)


def similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine_dist(v1, v2)) + 1) / 2     # 1 - cosine_dist = similarity.
    return score


def average_score(scores):
    score = sum(scores)
    if score > 0:
        return score/sum(s > 0 for s in scores)
    else:
        return 0


def clean(t):
    return ' '.join([t.strip() for t in t.split()])


def get_wc(sentences):
    count = 0
    for sentence in sentences:
        count += len(sentence.split())
    return count


if __name__ == '__main__':
    text = '''
WASHINGTON — President Donald Trump said Sunday that he is not considering firing special counsel Robert Mueller even as his administration was again forced to grapple with the growing Russia probe that has shadowed the White House for much of his initial year in office.

Trump returned to the White House from Camp David and was asked if he would consider triggering the process to dismiss Mueller, who is investigating whether the president’s Republican campaign coordinated with Russian officials during last year’s election.

The president answered: “No, I’m not.”

But he did add to the growing conservative criticism of Mueller’s move to gain access to thousands of emails sent and received by Trump officials before the start of his administration, yielding attacks from transition lawyers and renewing chatter that Trump may act to end the investigation.

“It’s not looking good. It’s quite sad to see that. My people were very upset about it,” Trump said. “I can’t imagine there’s anything on them, frankly. Because, as we said, there’s no collusion. There’s no collusion whatsoever.”

On Saturday, the general counsel for the transition group sent a letter to two congressional committees arguing Mueller’s investigators had improperly obtained thousands of transition records.

The investigators did not directly request the records from Trump’s still-existing transition group, Trump for America, and instead obtained them from the General Services Administration, a separate federal agency that stored the material, according to the group’s general counsel.

A spokesman for Mueller said the records were obtained appropriately.

“When we have obtained emails in the course of our ongoing criminal investigation, we have secured either the account owner’s consent or appropriate criminal process,” Peter Carr said.

But many Trump allies used the email issue as another cudgel with which to bash the probe’s credibility. Members of the conservative media and some congressional Republicans have begun to systematically question Mueller’s motives and credibility while the president himself called it a “disgrace” that some texts and emails from two FBI agents contained anti-Trump rhetoric. One of those agents was on Mueller’s team and has been removed.

Michael Caputo, a former Trump campaign aide, called the investigation an “attack on the presidency” and told CNN there are “more and more indications that the Mueller investigation is off the rails.”

The talk of firing Mueller has set off alarm bells among many Democrats, who warn it could trigger a constitutional crisis.

Some Republicans also advised against the move, including Sen. John Cornyn of Texas, who deemed the idea “a mistake.”

The rumor mill overshadowed the Republican tax plan, which is set to be voted on this week. Although Treasury Secretary Steve Mnuchin was doing a victory lap on the tax bill on the Sunday talk show circuit, he first had to field questions on CNN’s “State of the Union” about whether believed Trump would trigger the process to fire Mueller.

“I don’t have any reason to think that the president is going to do that, but that’s obviously up to him,” said Mnuchin.

Mnuchin added, “We have got to get past this investigation. It’s a giant distraction.” But he declined to elaborate on how he would want it to end. Marc Short, the White House director of legislative affairs, was also peppered with questions about Mueller’s fate during his own appearance on NBC’s “Meet the Press” and again urged a quick end to the investigation but insisted that Trump has not discussed firing Mueller.

“There’s no conversation about that whatsoever in the White House,” Short said.
    '''
    summarize = Summarizer()
    summarize.summarize(text)