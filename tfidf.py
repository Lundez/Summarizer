from sklearn.feature_extraction.text import TfidfVectorizer
import regex as re
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.stem.snowball import EnglishStemmer


def clean(t):
    """
    Some articles has a lot of whitespace by unknown reason.
    """
    return ' '.join([t.strip() for t in t.split()])


def clean_doc(text):
    text = sent_tokenize(text)
    text = [clean(t) for t in text]
    return text


def stem_words(doc, stemmer):
    result_doc = []
    for sen in doc:
        words = re.findall('\p{L}*', sen)
        words = [stemmer.stem(word) for word in words]
        result_doc.append(' '.join(words).strip())
    return result_doc


def parse_word_count(file, word_count=50, filetype='filename'):
    cleaned_text, maxi = parse(file, filetype)
    word_indexes = []
    size = 0
    idx = 0
    while idx < len(maxi) and size < word_count:
        word_index = maxi[idx]
        word_indexes.append(word_index)
        size += len(cleaned_text[word_index].split())
        idx += 1
    word_indexes.sort()
    summary = ""
    for idx in word_indexes:
        summary += " \n" + cleaned_text[idx]
    return summary.strip()


def parse_percentage(file, percent=0.1, filetype='filename'):
    cleaned_text, maxi = parse(file, filetype)
    text_size = len(' '.join(cleaned_text).split())
    word_indexes = []
    word_size = 0
    idx = 0
    while idx < len(maxi) and (word_size/text_size) < percent:
        word_index = maxi[idx]
        word_indexes.append(word_index)
        word_size += len(cleaned_text[word_index].split())
        idx += 1
    word_indexes.sort()
    summary = ""
    for idx in word_indexes:
        summary += " \n" + cleaned_text[idx]
    return summary.strip()


def parse(file, filetype='filename'):
    stemmer = EnglishStemmer()
    if filetype is 'filename':
        with open(file) as f:
            text = f.read().strip()
    else:
        text = file
    cleaned_text = clean_doc(text)
    stemmed_sentences = stem_words(cleaned_text, stemmer)
    tfidf_vect = TfidfVectorizer(stop_words='english')
    X = tfidf_vect.fit_transform(stemmed_sentences)
    tfidf = X.toarray()
    vector = tfidf.sum(0)
    vector = np.divide(vector, vector.max())

    vocab = tfidf_vect.get_feature_names()
    vocab2 = np.array(vocab)
    count = {}
    for sen in stemmed_sentences:
        sum_t = 0
        for word in [x for x in sen.split() if x in vocab2]:
            word_idx = list(vocab2).index(word)
            sum_t += X[0, word_idx]
        """
        Implementation of focus.
        for x in range(vector.shape[0]): 
            if vector[x] > 0.3:
                sum_t += vector[x]"""
        count[stemmed_sentences.index(sen)] = sum_t     # / len(sen.split()) <-- Gave worse results.
    maxi = sorted(count.keys(), key=count.get, reverse=True)
    return cleaned_text, maxi

if __name__ == "__main__":
    text = '''
    Cambodian leader Hun Sen on Friday rejected opposition parties' demands 
    for talks outside the country, accusing them of trying to ``internationalize'' 
    the political crisis. Government and opposition parties have asked 
    King Norodom Sihanouk to host a summit meeting after a series of post-election 
    negotiations between the two opposition groups and Hun Sen's party 
    to form a new government failed. Opposition leaders Prince Norodom 
    Ranariddh and Sam Rainsy, citing Hun Sen's threats to arrest opposition 
    figures after two alleged attempts on his life, said they could not 
    negotiate freely in Cambodia and called for talks at Sihanouk's residence 
    in Beijing. Hun Sen, however, rejected that. ``I would like to make 
    it clear that all meetings related to Cambodian affairs must be conducted 
    in the Kingdom of Cambodia,'' Hun Sen told reporters after a Cabinet 
    meeting on Friday. ``No-one should internationalize Cambodian affairs. 
    It is detrimental to the sovereignty of Cambodia,'' he said. Hun Sen's 
    Cambodian People's Party won 64 of the 122 parliamentary seats in 
    July's elections, short of the two-thirds majority needed to form 
    a government on its own. Ranariddh and Sam Rainsy have charged that 
    Hun Sen's victory in the elections was achieved through widespread 
    fraud. They have demanded a thorough investigation into their election 
    complaints as a precondition for their cooperation in getting the 
    national assembly moving and a new government formed. Hun Sen said 
    on Friday that the opposition concerns over their safety in the country 
    was ``just an excuse for them to stay abroad.'' Both Ranariddh and 
    Sam Rainsy have been outside the country since parliament was ceremonially 
    opened on Sep. 24. Sam Rainsy and a number of opposition figures have 
    been under court investigation for a grenade attack on Hun Sen's Phnom 
    Penh residence on Sep. 7. Hun Sen was not home at the time of the 
    attack, which was followed by a police crackdown on demonstrators 
    contesting Hun Sen's election victory. The Sam Rainsy Party, in a 
    statement released Friday, accused Hun Sen of being ``unwilling to 
    make any compromise'' on negotiations to break the deadlock. ``A meeting 
    outside Cambodia, as suggested by the opposition, could place all 
    parties on more equal footing,'' said the statement. ``But the ruling 
    party refuses to negotiate unless it is able to threaten its negotiating 
    partners with arrest or worse.'' 
    '''
    ret = parse_percent(text, filetype='content')
    print(ret)
    print(len(ret.split()))