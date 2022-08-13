# -*- coding: utf-8 -*-
"""data_preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zcIjhCecAXYQkxcACXGwR2hfkfsbI5fm
"""

## pip install datasets transformers[sentencepiece]
from datasets import load_dataset, DatasetDict,load_from_disk
#from zipfile import ZipFile
import os
from spellchecker import SpellChecker
from langdetect import detect
import re

def custom_load_dataset():
    reddit_dataset_raw = load_dataset("reddit", cache_dir='$HF_DATASETS_CACHE', download_mode='reuse_dataset_if_exists')
    print(reddit_dataset_raw)
    return reddit_dataset_raw

def remove_columns(dataset, column_names):
    '''
    dataset = raw dataset input
    column_names = list of strings, e.g. ["author", "body","subreddit_id","id", "normalizedBody"]
    '''
    reddit_dataset = dataset.remove_columns(column_names)
    print(reddit_dataset)
    return reddit_dataset

# Filter out all rows where either summary or content is blank
def filter_blanks(example):
    return example['content'] is not None and example['summary'] is not None

# Add columns for whitelist
def add_whitelisted(example):
    return {"whitelisted": example['subreddit'] in white_list}

# Add columns for "summary_len" and "content_len" columns
def compute_len(example):
    return {"summary_len": len(example["summary"].split()),
          "content_len": len(example["content"].split())}

###########################################################################################################################################

def remove_emoji(example):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return {"content" : emoji_pattern.sub(r'', example["content"]),
        "summary": emoji_pattern.sub(r'', example["summary"])}

# def remove_emoticons(example):
#     emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
#     return {"content" : emoticon_pattern.sub(r'', example["content"]),
#         "summary": emoticon_pattern.sub(r'', example["summary"])}

def remove_urls(example):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return {"content" : url_pattern.sub(r'', example["content"]),
        "summary": url_pattern.sub(r'', example["summary"])}

def remove_html(example):
    html_pattern = re.compile('<.*?>')
    return {"content" : html_pattern.sub(r'', example["content"]),
        "summary": html_pattern.sub(r'', example["summary"])}

def chat_words_map_dict_factory():
    chat_words_str = """
    AFAIK=As Far As I Know
    AFK=Away From Keyboard
    ASAP=As Soon As Possible
    ATK=At The Keyboard
    ATM=At The Moment
    A3=Anytime, Anywhere, Anyplace
    BAK=Back At Keyboard
    BBL=Be Back Later
    BBS=Be Back Soon
    BFN=Bye For Now
    B4N=Bye For Now
    BRB=Be Right Back
    BRT=Be Right There
    BTW=By The Way
    B4=Before
    B4N=Bye For Now
    CU=See You
    CUL8R=See You Later
    CYA=See You
    FAQ=Frequently Asked Questions
    FC=Fingers Crossed
    FWIW=For What It's Worth
    FYI=For Your Information
    GAL=Get A Life
    GG=Good Game
    GN=Good Night
    GMTA=Great Minds Think Alike
    GR8=Great!
    G9=Genius
    IC=I See
    ICQ=I Seek you (also a chat program)
    ILU=ILU: I Love You
    IMHO=In My Honest/Humble Opinion
    IMO=In My Opinion
    IOW=In Other Words
    IRL=In Real Life
    KISS=Keep It Simple, Stupid
    LDR=Long Distance Relationship
    LMAO=Laugh My Ass Off
    LOL=Laughing Out Loud
    LTNS=Long Time No See
    L8R=Later
    MTE=My Thoughts Exactly
    M8=Mate
    NRN=No Reply Necessary
    OIC=Oh I See
    PITA=Pain In The Ass
    PRT=Party
    PRW=Parents Are Watching
    ROFL=Rolling On The Floor Laughing
    ROFLOL=Rolling On The Floor Laughing Out Loud
    ROTFLMAO=Rolling On The Floor Laughing My Ass Off
    SK8=Skate
    STATS=Your sex and age
    ASL=Age, Sex, Location
    THX=Thank You
    TTFN=Ta-Ta For Now!
    TTYL=Talk To You Later
    U=You
    U2=You Too
    U4E=Yours For Ever
    WB=Welcome Back
    WTF=What The Fuck
    WTG=Way To Go!
    WUF=Where Are You From?
    W8=Wait
    """
    chat_words_map_dict = {}
    chat_words_list = []
    for line in chat_words_str.split("\n"):
        if line[0] != "":
            print(line.split("="))
            cw = line.split("=")[0]
            cw_expanded = line.split("=")[1]
            chat_words_list.append(cw)
            chat_words_map_dict[cw] = cw_expanded
    return chat_words_list, chat_words_map_dict

def chat_words_conversion(example):
    chat_words_list, chat_words_map_dict = chat_words_map_dict_factory()
    new_content = []
    new_summary = []

    for w in example["content"].split():
        if w.upper() in chat_words_list:
            new_content.append(chat_words_map_dict[w.upper()])
        else:
            new_content.append(w)
    content = " ".join(new_content)


    for w in example["summary"].split():
        if w.upper() in chat_words_list:
            new_summary.append(chat_words_map_dict[w.upper()])
        else:
            new_summary.append(w)
    summary = " ".join(new_summary)

    return {"content" : content,
        "summary": summary}

# return a boolean
def is_english(text):
    return detect(text) == 'en'

spell = SpellChecker()
def correct_spellings(example):
    corrected_content = []
    misspelled_content = spell.unknown(example["content"].split())
    for word in example["content"].split():
        if word in misspelled_content:
            corrected_content.append(spell.correction(word))
        else:
            corrected_content.append(word)
    content = " ".join(corrected_content)

    corrected_summary = []
    misspelled_summary = spell.unknown(example["summary"].split())
    for word in example["summary"].split():
        if word in misspelled_summary:
            corrected_summary.append(spell.correction(word))
        else:
            corrected_summary.append(word)
    summary = " ".join(corrected_summary)

    return {"content" : content,
        "summary": summary}

###########################################################################################################################################


# Split dataset into train, test, validation sets: 
def train_test_val_split(dataset, train_size, test_size, val_size):
    assert (train_size + test_size +  val_size) >= 1, "Sum of train_size, test_size, val_size must be 1"
    train_test_valid = dataset.train_test_split(shuffle = True, seed = 50, test_size=test_size+val_size)
    print(train_test_valid)

    # Split the 30% test + valid in half test, half valid
    test_valid = train_test_valid['test'].train_test_split(shuffle = True, seed = 50, test_size=(test_size/(test_size+val_size)))
    print(test_valid)

    # gather everyone if you want to have a single DatasetDict
    splitted_dataset = DatasetDict({
        'train': train_test_valid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})
    return splitted_dataset



if __name__=='__main__':
    white_list = ["relationships","AskReddit","relationship_advice","tifu","dating_advice","personalfinance","Advice","legaladvice","offmychest","loseit","jobs","self","BreakUps","askwomenadvice","dogs","running","pettyrevenge","needadvice","travel","Parenting","weddingplanning","Pets","Dogtraining","cats","AskDocs","college","GetMotivated","books","Cooking"]
    
    # Load dataset raw
    reddit_dataset_raw = custom_load_dataset()
 

    # Remove extra columns
    reddit_dataset = remove_columns(reddit_dataset_raw['train'], ["author", "body","subreddit_id","id", "normalizedBody"])

    # Clean blanks
    reddit_clean_blanks = reddit_dataset.filter(lambda x: filter_blanks(x))
    print("After filtering blanks len = ", len(reddit_clean_blanks))
    print(reddit_clean_blanks[0])


    # Add whitelisted column
    reddit_add_whitelisted = reddit_clean_blanks.map(add_whitelisted)
    print(reddit_add_whitelisted[0])
    # Filter rows that are in whitelist
    reddit_whitelisted = reddit_add_whitelisted.filter(lambda x: x["whitelisted"] is True)
    print("After keeping only whitelisted rows = ", len(reddit_whitelisted))
    print(reddit_whitelisted[0])



    # Add length columns for content and summary
    sample_add_len = reddit_whitelisted.map(compute_len)
    print(sample_add_len.num_rows)
    # Filter out rows with content length more than 512 tokens 
    sample_right_len = sample_add_len.filter(lambda x: x["content_len"] < 512)
    print(sample_right_len.num_rows)
    # Filter summaries that are less than 24 tokens or more than 48 tokens
    sample_right_len = sample_right_len.filter(lambda x: x["summary_len"] < 48 and x["summary_len"] > 24)
    print(sample_right_len.num_rows)
    
    print(sample_right_len[0])

    # Remove columns
    dataset = remove_columns(sample_right_len, ["whitelisted", "summary_len","content_len", 'subreddit'])
    print(dataset.num_rows)

###########################################################################################################################################

    # Remove duplicates: --> SKIP for now


    # Remove emoji, icon, URL, HTML tag
    dataset = dataset.map(remove_emoji)
    # dataset = dataset.map(remove_emoticons)
    dataset = dataset.map(remove_urls)
    dataset = dataset.map(remove_html)

    # Remove chat words
    dataset = dataset.map(chat_words_conversion)

    # Filter to get alpha, number, and english only
    dataset = dataset.filter(lambda x: is_english(x["content"]) == True)
    dataset = dataset.filter(lambda x: is_english(x["summary"]) == True)

    # Spelling correction
    dataset = dataset.map(chat_words_conversion)


###########################################################################################################################################


    # Split dataset into train, test, val sets: 
    dataset = train_test_val_split(dataset, train_size=0.7, test_size=0.15, val_size=0.15)
    print(dataset)

    # Save to disk
    dataset.save_to_disk("reddit_clean")
   
