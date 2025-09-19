import re
from copy import copy
import string
from unicodedata import normalize


def remove_crutch_words(input: str, crutch_word_list: list) -> str:
    punctuation = string.punctuation + "¡¿"
    input_copy = copy(input)
    split_input = input_copy.split()
    split_output = []
    for item in split_input:
        # Check without punctuation
        clean_item = item.lower().translate(str.maketrans("", "", punctuation))
        if clean_item not in crutch_word_list:
            split_output.append(item)

    return " ".join(split_output).strip()

def pad_punctuation(input: str) -> str:
    '''
    Insert whitespace between punctuation chars. Necessary for
    repeat ngram cleaning.
    '''
    output = re.sub(r'((?:\.{3})|[.,¡!¿?();:])', r' \1 ', input)
    output = re.sub(r'\s{2,}', ' ', output).strip()

    return output

def unpad_punctuation(input: str) -> str:
    output = re.sub(r' ((?:\.{3})|[.,!?);:])', r'\1 ', input)
    output = re.sub(r'([¡¿(]) ', r'\1', output)
    output = re.sub(r'\s{2,}', ' ', output).strip()

    return output

def remove_repeat_ngrams(input: str, max_order: int=2, verbose: bool=False) -> str: 
    """
    Remove repeated ngrams of up to max_order from the input string.
    """
    if not isinstance(input, str):
        return None
    if verbose: print(f"Input string: {input}")
    padded_string = pad_punctuation(input)
    split_string = padded_string.split(" ")
    for order in range(1, max_order+1):
        if verbose: print(f"\nCleaning ngrams of order {order}.")
        if verbose: print(f"Starting state: {split_string}")
        idx = 0
        next_ngram_lst = split_string[idx+order:idx+2*order]
        while next_ngram_lst: # check if we've reached the end
            curr_ngram_lst = split_string[idx:idx+order]
            next_ngram_lst = split_string[idx+order:idx+2*order]
            curr_ngram = " ".join(curr_ngram_lst).lower()
            next_ngram = " ".join(next_ngram_lst).lower()
            if verbose: print(f"\n  - Current ngram: {curr_ngram}\n  - Next ngram: {next_ngram}")
            if curr_ngram == next_ngram:
                del split_string[idx+order:idx+2*order] # delete next_ngram from split_string
                if verbose: print(f"  - State update: {split_string}")
            else:
                idx += 1
    output = unpad_punctuation(" ".join(split_string))
    if verbose: print(f"\nOutput string: {output}")
    
    return output

# Normalize text
def normalize_phrase(phrase:str)->str: 
    phrase_norm = re.sub('[%s]' % re.escape(string.punctuation), ' ', phrase)
    phrase_norm = phrase_norm.lower().replace("  ", " ")
    trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
    phrase_norm = normalize('NFKC', normalize('NFKD', phrase_norm).translate(trans_tab))
    return phrase_norm

# Delete stopwords
def remove_stopwords(text: str, stopwords: list):
    return ' '.join([word for word in text.split(' ') if word not in stopwords])


class CleanASRService:
    def __init__(self) -> None:
        return

    def execute(self, asr: str,stopwords_list: list=[], crutch_word_list: list=[], delete_stopwords: bool = False,verbose: bool=False) -> str:
        asr = remove_crutch_words(asr, crutch_word_list)
        asr = remove_repeat_ngrams(asr, max_order=2, verbose=verbose)
        if delete_stopwords:
            asr = remove_stopwords(asr,stopwords_list)
        asr = normalize_phrase(asr)
        return asr