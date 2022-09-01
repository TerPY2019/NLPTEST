import nltk
import itertools
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
from gensim.models.tfidfmodel import TfidfModel
# print(article)

# print(dictionary)

# computer_id = dictionary.token2id.get("computer")
# print(computer_id)

class bow:
    article = []
    def __init__(self,file) :
        self.flielist = file
    def ctoken(cls):
        cls.article = []
        for i in cls.flielist:
            f = open(i, "r")
            textfile = f.read()
            tokens = word_tokenize(textfile)
            lower_tokens = [t.lower()for t in tokens]
            alpha_only = [t for t in lower_tokens if t.isalpha()]
            no_stop = [t for t in alpha_only if t not in stopwords.words('english')]
            wordnet_lemmatizer = WordNetLemmatizer()
            lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stop]
            cls.article.append(lemmatized)
        return(cls.article)
    def dicttext(cls,stext):
        dictionary = Dictionary(cls.article)
        computer_id = dictionary.token2id.get(stext)
        if(computer_id == None):
            return("ไม่มีข้อความที่ใส่มา")
        else:
            return("มีข้อความนี้")
    def topfivebow(cls):
        dictionary = Dictionary(cls.article)
        corpus = [dictionary.doc2bow(a) for a in cls.article]
        total_word_count = defaultdict(int)
        for word_id, word_count in itertools.chain.from_iterable(corpus):
            total_word_count[word_id] += word_count
        sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1],reverse=True)
        showtext = "Top 5 Bow : \n"
        i = 0
        for word_id, word_count in sorted_word_count[:5]:
            i+=1
            showtext += str(i)+" : "+dictionary.get(word_id)+" "+str(word_count)+"\n"
        return(showtext)

    def topfivetfidf(cls):
        dictionary = Dictionary(cls.article)
        corpus = [dictionary.doc2bow(a) for a in cls.article]
        tfidf = TfidfModel(corpus)
        showtext1 = "Top 5  Tfidf : \n"
        c = 0
        for datatf in corpus:
            tfidf_weights = tfidf[datatf]
            sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
            for term_id, weight in sorted_tfidf_weights[:5]:
                c+=1
                showtext1 += str(c)+" : "+dictionary.get(term_id)+" "+str(weight)+"\n"
        return(showtext1)