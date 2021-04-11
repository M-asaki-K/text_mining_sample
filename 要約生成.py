# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:08:32 2021

@author: uni21
"""


import spacy
import neologdn
import re
import emoji
import mojimoji
import janome
from janome.tokenfilter import ExtractAttributeFilter
from janome.tokenfilter import POSKeepFilter
from janome.analyzer import Analyzer
from janome.analyzer import RegexReplaceCharFilter
from janome.analyzer import UnicodeNormalizeCharFilter
from janome.tokenizer import Tokenizer as JanomeTokenizer

class JapaneseCorpus:
    # ①
    def __init__(self):
        self.nlp = spacy.load('ja_ginza')
        self.analyzer = Analyzer(
            [UnicodeNormalizeCharFilter(), RegexReplaceCharFilter(r'[(\)「」、。]', ' ')],  # ()「」、。は全てスペースに置き換える
            JanomeTokenizer(),
            [POSKeepFilter(['名詞', '形容詞', '副詞', '動詞']), ExtractAttributeFilter('base_form')]  # 名詞・形容詞・副詞・動詞の原型のみ
        )

    # ②
    def preprocessing(self, text):
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\r', '', text)
        text = re.sub(r'\s', '', text)
        text = text.lower()
        text = mojimoji.zen_to_han(text, kana=True)
        text = mojimoji.han_to_zen(text, digit=False, ascii=False)
        text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
        text = neologdn.normalize(text)

        return text

    # ③
    def make_sentence_list(self, sentences):
        doc = self.nlp(sentences)
        self.ginza_sents_object = doc.sents
        sentence_list = [s for s in doc.sents]

        return sentence_list

    # ④
    def make_corpus(self):
        corpus = [' '.join(self.analyzer.analyze(str(s))) + '。' for s in self.ginza_sents_object]

        return corpus

class EnglishCorpus(JapaneseCorpus):
    # ①
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    # ②
    def preprocessing(self, text):
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\r', '', text)
        text = mojimoji.han_to_zen(text, digit=False, ascii=False)
        text = mojimoji.zen_to_han(text, kana=True)
        text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
        text = neologdn.normalize(text)        

        return text

    # ④
    def make_corpus(self):
        corpus = []
        for s in self.ginza_sents_object:
            tokens = [str(t) for t in s]
            corpus.append(' '.join(tokens))

        return corpus
    
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words

# algorithms
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer

algorithm_dic = {"lex": LexRankSummarizer(), "tex": TextRankSummarizer(), "lsa": LsaSummarizer(),\
                 "kl": KLSummarizer(), "luhn": LuhnSummarizer(), "redu": ReductionSummarizer(),\
                 "sum": SumBasicSummarizer()}

def summarize_sentences(sentences, sentences_count=3, algorithm="lex", language="japanese"):
    # ①
    if language == "japanese":
        corpus_maker = JapaneseCorpus()
    else:
        corpus_maker = EnglishCorpus()
    preprocessed_sentences = corpus_maker.preprocessing(sentences)
    preprocessed_sentence_list = corpus_maker.make_sentence_list(preprocessed_sentences)
    corpus = corpus_maker.make_corpus()
    parser = PlaintextParser.from_string(" ".join(corpus), Tokenizer(language))

    # ②
    try:
        summarizer = algorithm_dic[algorithm]
    except KeyError:
        print("algorithm name:'{}'is not found.".format(algorithm))
    summarizer.stop_words = get_stop_words(language)
    summary = summarizer(document=parser.document, sentences_count=sentences_count)

    # ③
    if language == "japanese":
        return "".join([str(preprocessed_sentence_list[corpus.index(sentence.__str__())]) for sentence in summary])
    else:
        return " ".join([sentence.__str__() for sentence in summary])

text = """
好きな文章をここに書いてください。
"""
sentences_count = 3
algorithm = "lex"
language="japanese"
sum_sentences = summarize_sentences(text, sentences_count=sentences_count, algorithm=algorithm, language=language)
print(sum_sentences)
