import tqdm
import os
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spacy.lang.pt.stop_words import STOP_WORDS
from wordcloud.wordcloud import STOPWORDS
import re
import string
import unidecode
import pickle
from collections import Counter


def getStopwords() -> list:
    portuguese_english_stopwords = []
    portuguese_english_stopwords.extend([unidecode.unidecode(word.lower().replace(" ", "")) for word in
                                         nltk.corpus.stopwords.words('portuguese')])

    portuguese_english_stopwords.extend([unidecode.unidecode(word.lower().replace(" ", "")) for word in
                                         nltk.corpus.stopwords.words('english')])
    portuguese_english_stopwords.extend(
        [unidecode.unidecode(word.lower().replace(" ", "")) for word in STOPWORDS])
    portuguese_english_stopwords.extend([unidecode.unidecode(word.lower().replace(" ", "")) for word in STOP_WORDS])

    with open('stopwords-pt.txt', 'r', encoding='utf-8') as words:
        portuguese_english_stopwords.extend([unidecode.unidecode(word.lower().replace(" ", "")) for word in words])

    stopwords = '''
                a, agora, ainda, alguém, algum, alguma, algumas, alguns, ampla, amplas, amplo, amplos, ante, antes,
                ao, aos, após, aquela, aquelas, aquele, aqueles, aquilo, as, até, através, cada, coisa, coisas, com,
                como, contra, contudo, da, daquele, daqueles, das, de, dela, delas, dele, deles, depois, dessa,
                dessas, desse, desses, desta, destas, deste, destes, deve, devem, devendo, dever, deverá, deverão,
                deveria, deveriam, devia, deviam, disse, disso, disto, dito, diz, dizem, do, dos, e, é, ela, elas,
                ele, eles, em, enquanto, entre, era, essa, essas, esse, esses, esta, está, estamos, estão, estas,
                estava, estavam, estávamos, este, estes, estou, eu, fazendo, fazer, feita, feitas, feito, feitos,
                foi, for, foram, fosse, fossem, grande, grandes, há, isso, isto, já, la, lá, lhe, lhes, lo, mas, me,
                mesma, mesmas, mesmo, mesmos, meu, meus, minha, minhas, muita, muitas, muito, muitos, na, não, nas,
                nem, nenhum, nessa, nessas, nesta, nestas, ninguém, no, nos, nós, nossa, nossas, nosso, nossos, num,
                numa, nunca, o, os, ou, outra, outras, outro, outros, para, pela, pelas, pelo, pelos, pequena,
                pequenas, pequeno, pequenos, per, perante, pode, pude, podendo, poder, poderia, poderiam, podia, podiam,
                pois, por, porém, porque, posso, pouca, poucas, pouco, poucos, primeiro, primeiros, própria, próprias,
                próprio, próprios, quais, qual, quando, quanto, quantos, que, quem, são, se, seja, sejam, sem, sempre,
                sendo, será, serão, seu, seus, si, sido, só, sob, sobre, sua, suas, talvez, também, tampouco, te, tem,
                tendo, tenha, ter, teu, teus, ti, tido, tinha, tinham, toda, todas, todavia, todo, todos, tu, tua, tuas, 
                tudo, última, últimas, último, últimos, um, uma, umas, uns, vendo, ver, vez, vindo, vir, vos, vós a,
                able, about, across, after, all, almost, also, am, among, an, and, any, are, as, at, be, because, been,
                but, by, can, cannot, could, dear, did, do, does, either, else, ever, every, for, from, get, got, had,
                has, have, he, her, hers, him, his, how, however, i, if, in, into, is, it, its, just, least, let, like,
                likely, may, me, might, most, must, my, neither, no, nor, not, of, off, often, on, only, or, other, our,
                own, rather, said, say, says, she, should, since, so, some, than, that, the, their, them, then, there,
                these, they, this, tis, to, too, twas, us, wants, was, we, were, what, when, where, which, while, who,
                whom, why, will, with, would, yet, you, your
                '''

    tokenizer = nltk.RegexpTokenizer(r"\w+")
    portuguese_english_stopwords.extend(
        [unidecode.unidecode(palavra.lower().replace(" ", "")) for palavra in tokenizer.tokenize(stopwords)])

    stopwords = ['haha', 'ai', 'acho', 'nan', 'fica', 'vao', 'quer', 'queria', 'querer', 'achei', 'fica', 'ficou',
                 'deixa', 'deixou', 'pra', 'to', 'vc', 'tá', 'pq', 'tô', 'ta', 'mt', 'pro', 'né', 'eh', 'tbm', 'ja',
                 'ah', 'vcs', 'hj', 'so', 'mto', 'agr', 'oq', 'la', 'tou', 'td', 'voce', 'ne', 'obg', 'tb', 'pra',
                 'to', 'vc', 'tá', 'pq', 'tô', 'ta', 'mt', 'pro', 'né', 'eh', 'tbm', 'ja', 'ah', 'vcs', 'hj', 'so',
                 'mto', 'agr', 'oq', 'la', 'tou', 'td', 'voce', 'ne', 'obg', 'tb', 'pra', 'vc', 'pra', 'to', 'os',
                 'rappi', 'vcs', 'nao', 'pq', 'mim', 'ai', 'ta', 'ja', 'ter', 'fazer', 'lá', 'deu', 'dado', 'então',
                 'vou', 'vai', 'veze', 'ficar', 'tá', 'apena', 'apenas', 'melhor', 'cara', 'gente', 'casa', 'pessoa',
                 'tocada', 'tava', 'falar', 'serum', 'gt', 'bts', 'ia', 'preciso', 'vox', 'fico', 'sair', 'tomar',
                 'quase', 'conta', 'al', 'falar', 'falando', 'amo', 'amor'
                 ]

    portuguese_english_stopwords.extend(
        [unidecode.unidecode(palavra.lower().replace(" ", "")) for palavra in stopwords])

    portuguese_english_stopwords = list(set(portuguese_english_stopwords))
    portuguese_english_stopwords.sort()

    return portuguese_english_stopwords


def removeStopwords(text, stopwords) -> str:
    a = [i for i in text.split() if i not in stopwords]
    return ' '.join(a)


def normalize(text) -> str:
    text = text.lower().strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub("@\S+", "", text)
    text = re.sub("\$", "", text)
    text = re.sub("https?:\/\/.*[\r\n]*", "", text)
    text = re.sub(r'https?:\/\/[\r\n],"[\r\n]"', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub('[^a-zà-ù ]', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub('#', ' ', text)
    text = re.sub(r'.*kkk*\S*', ' haha ', text)
    text = re.sub(r'.*(rs)(rs)+\S*', ' haha ', text)
    text = re.sub(r'.*(ha|hu)\S*(ha|hu)+\S*', ' haha ', text)
    text = re.sub(r'[^\u0061-\u007A\u0020]', ' ', text)
    return text


def categorize(token) -> str:
    if token.startswith('J'):
        return wordnet.ADJ
    elif token.startswith('V'):
        return wordnet.VERB
    elif token.startswith('N'):
        return wordnet.NOUN
    elif token.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


class NaturalLanguageProcessing:

    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt', quiet=True)
        self.wl = WordNetLemmatizer

    def lemmatize(self, text) -> str:
        token = word_tokenize(text)
        word_pos_tags = nltk.pos_tag(token)
        a = [self.wl.lemmatize(tag[0], self.obter_pos_tag(tag[1])) for idx, tag in
             enumerate(word_pos_tags)]
        return " ".join(a)

    def preProcessing(self, cleaned_text) -> str:
        stopwords = self.getStopwords()

        cleaned_text = self.normalize(cleaned_text)
        cleaned_text = self.removeStopwords(cleaned_text, stopwords)
        cleaned_text = self.lematize(cleaned_text)

        return cleaned_text

    def createBDText(self, cvs_input, cvs_output, directory):
        usecols = ['created_at', 'id', 'id_screen_name', 'screen_name', 'full_text']
        for user in tqdm.tqdm(cvs_input):
            df = pd.read_csv(user, sep=';',
                             usecols=usecols,
                             index_col='created_at',
                             parse_dates=['created_at']).drop_duplicates(subset='id',
                                                                         keep='first',
                                                                         inplace=False).sort_index(inplace=False)

            df['full_text_clean'] = df['full_text'].apply(str).apply(lambda x: self.processamentoGeral(x))
            df['full_text_len'] = df['full_text_clean'].astype(str).apply(len)
            df['full_text_word_count'] = df['full_text_clean'].apply(lambda x: len(str(x).split()))

            df.drop_duplicates(subset='id', keep='first', inplace=True)
            df.rename_axis('created_at').reset_index(inplace=True)

            if not os.path.isfile(directory + cvs_output):
                df.to_csv(directory + cvs_output, sep=';', encoding='utf-8')
            else:
                df.to_csv(directory + cvs_output, sep=';', encoding='utf-8', mode='a', header=False)

    def generateWordFrequencyDictionary(self, periodo, diretorio):
        """_summary_

        Args:
            periodo (str): _description_
            diretorio (str): _description_
        """
        arquivoDepressao = f'{diretorio}twitterTextos_tweets_depressao_{periodo}.csv'
        arquivoControle = f'{diretorio}twitterTextos_tweets_controle_{periodo}.csv'

        chunksize = 10 ** 5

        counterDepressao = Counter()
        counterControle = Counter()

        dfDeprpessao = pd.read_csv(arquivoDepressao, sep=';', encoding='utf-8', usecols=['full_text_clean'],
                                   iterator=True, chunksize=chunksize)
        dfControle = pd.read_csv(arquivoControle, sep=';', encoding='utf-8', usecols=['full_text_clean'], iterator=True,
                                 chunksize=chunksize)

        for i, df in enumerate(dfDeprpessao):
            print(f'{periodo} dfDeprpessao parte {i + 1}')
            counterDepressao.update(word_tokenize(' '.join(df['full_text_clean'].astype(str).tolist())))

        with open(f'{periodo}_counterDepressao.pickle', 'wb') as f:
            pickle.dump(counterDepressao, f, protocol=pickle.HIGHEST_PROTOCOL)

        for i, df in enumerate(dfControle):
            print(f'{periodo} dfControle parte {i + 1}')
            counterControle.update(word_tokenize(' '.join(df['full_text_clean'].astype(str).tolist())))

        with open(f'{periodo}_counterControle.pickle', 'wb') as f:
            pickle.dump(counterControle, f, protocol=pickle.HIGHEST_PROTOCOL)
