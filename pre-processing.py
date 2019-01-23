import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps =PorterStemmer()

negations_dic={"isn't":"is not", "aren't":"are not", "wasn't":"was not","weren't":"were not",
               "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
               "wouldn't":"would not","don't":"do not","doesn't":"does not","didn't":"did not",
               "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
               "mustn't":"must not"}
neg_pattern =re.compile(r'\b('+'|'.join(negations_dic.keys())+r')\b')

stop_words={
               "a":'', "about":'', "above":'', "across":'', "after":'', "afterwards":'', "again":'',"am":'', "an":'',
               "and":'', "are":'', "at":'', "back":'',"be":'', "been":'', "being":'', "below":'', "beside":'',
               "besides":'',"between":'', "beyond":'', "bottom":'', "by":'', "call":'',"can":'',"could":'', "down":'',
               "did":'',"do":'',"does":'',"doing":'',"due":'', "eg":'', "either":'',"else":'', "elsewhere":'', "etc":'',"for":'',
               "few":'', "from":'', "front":'', "get":'', "had":'', "has":'',
               "have":'',"having":'', "he":'', "he'd":'',"he'll":'',"he's":'', "her":'', "here":'', "here's":'', "hers":'',
               "herself":'', "him":'', "himself":'', "his":'', "how":'',"how's":'',"i":'',"i'd":'',"i'll":'',"i'm":'',"i've":'',"if":'',"in":'',"into":'',
               "is":'', "it":'',"it's":'', "its":'', "itself":'',"me":'', "might":'', "mine":'', "must":'', "my":'', "myself":'',
               "of":'', "on":'',"only":'', "or":'', "other":'', "others":'',"our":'', "ours":'', "ourselves":'', "out":'', "over":'', "own":'',
               "she":'',"she'd":'',"she'll":'',"she's":'', "somehow":'', "someone":'',
               "such":'', "take":'', "than":'', "that":'', "the":'',"that's":'',"their":'',"theirs":'', "them":'', "themselves":'',"then":'', "there":'',"there's":'',
               "these":'', "they":'',"they'd":'',"they'll":'',"they're":'',"they've":'', "this":'', "those":'', "though":'',"to":'', "too":'',
               "under":'', "until":'', "up":'', "very":'', "via":'', "was":'', "we":'',"we'd":'',"we'll":'',"we're":'',"we've":'',
               "were":'', "what":'',"what's":'', "when":'',"when's":'', "where":'', "where's":'',
               "which":'', "while":'',"whither":'', "who":'',"who's":'', "whom":'', "whose":'', "why":'',"why's":'', "will":'', "with":'', "would":'',"yet":'',
               "you":'',"you'd":'',"you'll":'',"you're":'',"you've":'', "your":'', "yours":'', "yourself":'', "yourselves":'',
}
stop_pattern =re.compile(r'\b('+'|'.join(stop_words.keys())+r')\b')

helping_verb_dic={"i'm":"i am","he's":"he is","she's":"she is","it's":"it is",
                  "i've":"i have","we're":"we are","you're":"you are","they're":"they are",
                  "i'll":"i will","you'll":"you will","we'll":"we will","they'll":"they will",
                  "he'll":"he will","she'll":"she will","it'll":"it will"}

helping_pattern =re.compile(r'\b('+'|'.join(helping_verb_dic.keys())+r')\b')

def pre_process(text):
    soup = BeautifulSoup(text, "html.parser")
    souped = soup.get_text()
    lower_case = souped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    helping_handled = helping_pattern.sub(lambda x: helping_verb_dic[x.group()], neg_handled)
    stop_handled = stop_pattern.sub(lambda x: stop_words[x.group()], helping_handled)

    # convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    repitition_handled = re.sub(r'(.)\1+', r'\1\1', stop_handled)
    # print(repitition_handled)

    letters_only = re.sub(r'\d+', '', repitition_handled)
    return(letters_only)


