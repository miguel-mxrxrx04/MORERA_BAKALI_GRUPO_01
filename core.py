import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from typing import List

#Modulo 1
def preprocess_post(text: str):

    """
    Versión optimizada de la función de limpieza de texto.
    Aplica reemplazos de patrones y realiza la tokenización,
    filtrando stopwords y caracteres no alfanuméricos.
    """

        # 1. Precompilar patrones regex (más eficiente)
    HTML_PATTERN = re.compile(r'<[^>]*>')
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    MENTION_PATTERN = re.compile(r'@\w+|#\w+')
    SPECIAL_CHARS = re.compile(r'[^\w\s]')

    # 2. Cargar recursos NLTK una sola vez
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

        #Verifica si el texto es NaN
    if pd.isna(text):
        return ""
    
    # # Limpieza general del texto
    text = (HTML_PATTERN.sub(' ', text.lower())
            .replace('\n', ' ')
            .replace('\t', ' '))
    
    text = URL_PATTERN.sub('', text)
    text = MENTION_PATTERN.sub('', text)
    text = SPECIAL_CHARS.sub(' ', text)
    
    # Tokenizar y limpiar en una sola pasada
    tokens = [
        lemmatizer.lemmatize(token)
        for token in word_tokenize(text)
        if token not in stop_words and token.isalnum()
    ]
    
    return ' '.join(tokens)
#Modulo 2
def classify_subredit(text):
    pass

#Modulo 3
def find_subreddit_mentions(text: str) -> List[str]|str:
    subreddit_mention_pattern = re.compile(r'/r/[A-Za-z0-9]')
    subreddit_mentions = re.findall(subreddit_mention_pattern, text)
    return subreddit_mentions

def url_extraction(text: str) -> List[str]|str:
    url_pattern = patron_url = re.compile(
    r'(https?://(?:www\.|(?!www))[^\s]+)|'      # URLs con http:// o https://
    r'(www\.[^\s]+)|'                           # URLs que empiezan con www.
    r'([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'           # URLs sin protocolo
    )
    urls = re.findall(text, url_pattern)
    return urls

def phone_number_extraction(text: str) -> List[str]|str:
    phone_number_pattern = re.compile(
    r'(\+?\d{1,3}[-.\s]?)?'      # Prefijo internacional opcional
    r'(\(?\d{3}\)?[-.\s]?)?'     # Código de área opcional con o sin paréntesis
    r'(\d{3}[-.\s]?\d{4})'       # Número de teléfono principal
    r'|(\+?\d{1,3}[-.\s]?)?'     # Prefijo internacional opcional para números de 3 grupos
    r'(\d{3}[-.\s]\d{3}[-.\s]\d{3})'  # Formato tipo "654 231 235"
    )
    phone_numbers = re.findall(text, phone_number_pattern)
    return phone_numbers

def dates_extraction(text: str) -> List[str]|str:
    dates_pattern = re.compile(r'')
    dates = re.findall(text, dates_pattern)
    return dates

def code_extraction(text: str) -> List[str]|str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    html_code = soup.prettify()
    return html_code

#Modulo 4
def sentiment_analysis(text: str):
    pass

# Modulo 5
def post_summarisation(text: str):
    pass

#Modulo 6
def texts_distance(text1: str, text2: str):
    pass 
