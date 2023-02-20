#Add spaces in the most likely place given string


import enchant
dict = enchant.Dict("en_US")

import nltk
import numpy as np
import pandas as pd
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from wordfreq import word_frequency

tags_df = pd.read_pickle("posdf.pkl")       #dataframe of probabilities for neighboring parts of speech

def split_text(text, cache = []):
    if not text:
        return 1, []
    best_p, best_split = -1, []
    for i in range(1, len(text) + 1):
        word, remainder = text[:i], text[i:]
        #ipdb.set_trace(context=6)
        if (len(word) > 1) or (word == "a") or (word == "i"):
          f = word_frequency(word,"en")
          if f > 5e-6:
             if cache == []:
                 curr_pos = nltk.pos_tag([word])
                 p = tags_df.loc['.', curr_pos[0][1]]
             else:
                 curr_pos = nltk.pos_tag([word])
                 #ipdb.set_trace(context=6)
                 p = tags_df.loc[cache[-1][1]],[curr_pos[0][1]]
             p = p*f
             if remainder != "":
                 remainder_p, remainder = split_text(remainder, cache + curr_pos)
                 p *= remainder_p
             if p > best_p:
                 best_p = p
                 #ipdb.set_trace(context=6)
                 if remainder != "":
                     best_split = curr_pos + remainder
                 else:
                     best_split = curr_pos
    cache = (best_p, best_split)
    return cache


def add_spaces(text, cache = []):
    if cache != []:
        cache = nltk.pos_tag(cache)
    words = split_text(text, cache)
    answer = ""
    temp = []
    #ipdb.set_trace(context=6)
    if words[0] < 1e-40:
        answer = "-".join(text)
        temp.append(answer)
    else:
        for i in words[1]:
            answer = answer + i[0] + " "
            temp.append(i[0])
        answer = answer[:-1]
    return answer, temp
