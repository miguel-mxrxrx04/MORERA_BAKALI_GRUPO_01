import pandas as pd
from typing import List
import time
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



#Modulo 1
def preprocess_post(text: str):

    """
    Versión optimizada de la función de limpieza de texto.
    Aplica reemplazos de patrones y realiza la tokenización,
    filtrando stopwords y caracteres no alfanuméricos.
    """
    HTML_PATTERN = re.compile(r'<[^>]*>')
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    MENTION_PATTERN = re.compile(r'@\w+|#\w+')
    SPECIAL_CHARS = re.compile(r'[^\w\s]')

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

# Modulo 2
def classify_subredit(text):
    pass

# Modulo 3
from typing import List
import re

def find_subreddit_mentions(text: str) -> List[str]:
    """
    Extrae menciones a subreddits del texto.

    Parámetros:
    - text (str): Texto del cual se extraerán las menciones a subreddits.

    Retorna:
    - List[str]: Lista de menciones de subreddits encontradas.
    """
    # Patrón regex para menciones a subreddits (e.g., /r/Python)
    subreddit_mention_pattern = re.compile(r'/r/[A-Za-z]{1}[A-Za-z0-9]{2,22}')
    subreddit_mentions = subreddit_mention_pattern.findall(text)
    return subreddit_mentions

def url_extraction(text: str) -> List[str]:
    """
    Extrae URLs del texto
    """
    url_pattern = re.compile(
        r'(https?://(?:www\.|(?!www))[^\s]+)|'      # URLs con http:// o https://
        r'(www\.[^\s]+)|'                           # URLs que empiezan con www.
        r'([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'          # URLs sin protocolo
    )
    # Corregimos el orden: primero va el patrón, luego el texto
    urls = re.findall(url_pattern, text)
    
    # Como findall con grupos retorna tuplas, necesitamos procesar el resultado
    # para obtener solo las URLs válidas
    extracted_urls = []
    for url_tuple in urls:
        # Tomar la primera URL no vacía de cada tupla
        url = next((u for u in url_tuple if u), None)
        if url:
            extracted_urls.append(url)
    
    return extracted_urls

def phone_number_extraction(text: str) -> List[str]|str:
    """
    Extrae números de teléfono del texto, incluyendo formatos estándar y grupos de 3 dígitos.

    Parámetros:
    - text (str): Texto del cual se extraerán los números de teléfono.

    Retorna:
    - List[str]: Lista de números de teléfono encontrados.
    """
    # Definir el patrón regex sin grupos de captura internos
    phone_number_pattern = re.compile(
        r'(?:\+?\d{1,3}[-.\s]?)?'      # Prefijo internacional opcional
        r'(?:\(?\d{3}\)?[-.\s]?)?'     # Código de área opcional con o sin paréntesis
        r'\d{3}[-.\s]?\d{4}'            # Número de teléfono principal
        r'|'                            # Alternativa
        r'(?:\+?\d{1,3}[-.\s]?)?'      # Prefijo internacional opcional para formatos alternativos
        r'\d{3}[-.\s]\d{3}[-.\s]\d{3}'  # Formato tipo "654 231 235"
    )
    phone_numbers = phone_number_pattern.findall(text)
    return [number.strip() for number in phone_numbers if number.strip()]

def dates_extraction(text: str) -> List[str]:
    """
    Extrae todas las fechas del texto en diferentes formatos utilizando una expresión regular.

    Parámetros:
    - text (str): Texto del cual se extraerán las fechas.

    Retorna:
    - List[str]: Lista de fechas encontradas.
    """
    # Definir el patrón regex para extraer fechas en varios formatos
    dates_pattern = re.compile(
        r'(?<!\w)'  # Asegura que la fecha no esté precedida por una letra
        r'('
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|'                            # DD/MM/YYYY, MM-DD-YY, etc.
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b|'  # Month DD, YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'                             # YYYY/MM/DD, YYYY-MM-DD
        r')'
    )
    dates = dates_pattern.findall(text)
    return dates

def code_extraction(text: str) -> List[str]|str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    html_code = soup.prettify()
    return html_code

# Modulo 4
def sentiment_analysis(text: str):
    pass

# Modulo 5
def post_summarisation(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    import string
    """
    Genera un resumen extractivo usando frecuencia de palabras.
    text: Texto original (multi-línea) del que se extraerá el resumen.
    summary_sentences: Número de oraciones que quieres en tu resumen.
    """
    # Separar en oraciones
    sentences = sent_tokenize(text)
    
    # Separar en palabras cada oración para calcular frecuencia
    stop_words = set(stopwords.words('english'))  # Ajusta idioma si es necesario
    word_frequencies = {}
    
    for sentence in sentences:
        # Tokenizar en palabras y limpiar signos de puntuación
        words = word_tokenize(sentence)
        for word in words:
            # Conviertes todo a minúscula y quitas signos de puntuación
            word = word.lower()
            if word not in stop_words and word not in string.punctuation:
                word_frequencies[word] = word_frequencies.get(word, 0) + 1
    
    # Calcular la frecuencia máxima para normalizar
    max_frequency = max(word_frequencies.values()) if word_frequencies else 1
    
    # Asignar puntuación a cada oración según la frecuencia de sus palabras
    sentence_scores = {}
    for sentence in sentences:
        sentence_scores[sentence] = 0
        words_in_sentence = word_tokenize(sentence.lower())
        for word in words_in_sentence:
            if word in word_frequencies:
                # Se suma la frecuencia normalizada de la palabra
                sentence_scores[sentence] += word_frequencies[word] / max_frequency

    # Ordenar oraciones según la puntuación y escoger las mejores
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    final_sentences = len(sentences) // 2
    summary = " ".join(ranked_sentences[:final_sentences])

    return summary


#Modulo 6
def texts_distance(text1: str, text2: str):
    pass 
