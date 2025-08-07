from backend.generative_engine_client import generative_engine_summary
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def extractive_summary(text, sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences)
    return " ".join(str(sentence) for sentence in summary)

def get_resume_summary(text):
    try:
        return generative_engine_summary(text)
    except Exception as e:
        return extractive_summary(text)
