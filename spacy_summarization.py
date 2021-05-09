# NLP Pkgs
from langdetect import detect
import spacy

# Pkgs for Normalizing Text
from spacy.lang.en.stop_words import STOP_WORDS
# Import Heapq for Finding the Top N Sentences
from heapq import nlargest


dict_model_nlp = {
    "en": spacy.load("en_core_web_sm"), # english
    "fr": spacy.load("fr_core_news_sm"), # french
    "zh-cn": spacy.load("zh_core_web_sm"), # chinese
    "da": spacy.load("da_core_news_sm"), # danish
    "de": spacy.load("de_core_news_sm"), # german
    "nl": spacy.load("nl_core_news_sm"), # dutch
    "el": spacy.load("el_core_news_sm"), # greek
    "it": spacy.load("it_core_news_sm"), # italian
    "ja": spacy.load("ja_core_news_sm"), # japanese
    "no": spacy.load("nb_core_news_sm"), # norwegian
    "pl": spacy.load("pl_core_news_sm"), # polish
    "pt": spacy.load("pt_core_news_sm"), # portuguese
    "ro": spacy.load("ro_core_news_sm"), # romanian
    "es": spacy.load("es_core_news_sm") # spanish
}
nlp_tt = spacy.load("xx_ent_wiki_sm")


def text_summarizer(raw_docx):
    raw_text = raw_docx
    lang = detect(raw_text)
    if lang in dict_model_nlp:
        nlp = dict_model_nlp[lang]
    else:
        nlp = nlp_tt
    docx = nlp(raw_text)
    stopwords = list(STOP_WORDS)
    # Build Word Frequency # word.text is tokenization in spacy
    word_frequencies = {}
    for word in docx:
        if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / maximum_frequncy
    # Sentence Tokens
    sentence_list = [sentence for sentence in docx.sents]

    # Sentence Scores
    sentence_scores = {}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]

    summarized_sentences = nlargest(7, sentence_scores, key=sentence_scores.get)
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)
    return summary
