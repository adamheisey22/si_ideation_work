
import pandas as pd
import numpy as np
from negex import *
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import RegexpTokenizer
regexp = RegexpTokenizer('\w+')


# def extract_between_strings(text, start, end):
#     start_index = text.find(start)
#     if start_index == -1:
#         return ''
#
#     end_index = text.find(end, start_index + len(start))
#     if end_index == -1:
#         return ''
#
#     return text[start_index + len(start):end_index]
#
# def cssr_scrub(df, data_column):
#     df['CSSRS_FORM'] = df[data_column].apply(lambda x: extract_between_strings(x, 'Within normal limits', 'Medical History'))
#     df[data_column] = df[data_column].apply(lambda x: x.replace(extract_between_strings(x, 'Within normal limits', 'Medical History'), ' '))
#     return df


def cssr_txt(note_text):

    if 'Within normal limits' in note_text:
        start = 'Within normal limits'
        end = 'Medical History'

        s = note_text

        cs = s[s.find(start)+len(start):s.rfind(end)]
        return cs

    else:
        return ' '

def cssr_scrub(df, data_column):
    ''' scrub out the cssrs form to do something with '''

    df['CSSRS_FORM'] = df[data_column].apply(lambda x: cssr_txt(x))

    df[f'{data_column}'] = df[data_column].apply(lambda x: x.replace(cssr_txt(x), ' '))

    return df

# def clean_html(df, data_column):
#     '''Extract text from HTML'''
#
#     df['RAW_TEXT'] = df[data_column].astype(str).apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
#     df['TEXT_LINES'] = df['RAW_TEXT'].apply(lambda x: nltk.sent_tokenize(x))
#     df['NOTE_TEXT'] = df['TEXT_LINES'].apply(lambda x: ' '.join(x))
#     df['NOTE_TEXT'] = df['NOTE_TEXT'].str.strip()
#     df['NOTE_TEXT'] = df['NOTE_TEXT'].str.splitlines().apply(lambda x: ', '.join(x))
#     df['NOTE_TEXT'] = df['NOTE_TEXT'].str.replace('\s+', ' ')
#
#     return df

def clean_html(df, data_column):
    '''extract the text from html'''

    #df['RAW_TEXT'] = df['blob_contents_decoded_no_tag'].astype(str).apply(lambda x: html_to_text(x))
    df['RAW_TEXT'] = df[data_column].astype(str).apply(lambda x: remove_tags(x))
    # df['RAW_TEXT'] = df['blob_contents_decoded_no_tag'].astype(str).apply(lambda x: support_functions.remove_this(x))
    # #df['RAW_TEXT'] = df['blob_contents_decoded_no_tag'].astype(str).apply(lambda x: BeautifulSoup(x, "html").get_text())
    df['TEXT_LINES']=df['RAW_TEXT'].astype(str).apply(lambda x: nltk.tokenize.sent_tokenize(x))
    df['NOTE_TEXT'] = df['TEXT_LINES'].apply(lambda x: ' '.join([item for item in x]))
    df['NOTE_TEXT'] = df['NOTE_TEXT'].apply(lambda x: x.strip())
    df['NOTE_TEXT'] = df['NOTE_TEXT'].apply(lambda x: x.splitlines())
    df['NOTE_TEXT'] = df['NOTE_TEXT'].apply(lambda x: ', '.join([item for item in x]))
    df['NOTE_TEXT'] = df['NOTE_TEXT'].apply(lambda x:  ' '.join(x.split()))

    return df

def scrub_disclaimer(df, data_column):
    replacements = {
        'SUICIDE AWARENESS Warning Signs can include personal situations, thoughts, images, thinking types, mood and behavior': ' ',
        'In Case of Emergency Dial 911 or nearest Emergency Room National Suicide Prevention Life Line 1-800-273-TALK or 1-800-273-8255 http://www.suicidepreventionlifeline.org': ' ',
        'Reduce potential for lethal means Family and Friends should assist in securing or removing from the house or car all means of committing suicide impulsively': ' ',
        'National Alliance on Mental Illness 1-801-323-9900 or 1-877-230-6264 http://www.nami.org Substance Abuse and Mental Health Services Administration 1-877-726-4727 http://www.samhsa.gov/suicide-prevention': ' '
    }

    for phrase, replacement in replacements.items():
        df[f'{data_column}'] = df[data_column].replace(phrase, replacement, regex=True)

    return df


# def scrub_disclaimer(df, data_column):
#
#     ''' This function replaces the following phrases of the disclosure statement '''
#
#     df[f'{data_column}'] = df[data_column].apply(lambda x: x.replace('SUICIDE AWARENESS Warning Signs can include personal situations, thoughts, images, thinking types, mood and behavior', ' '))
#     df[f'{data_column}'] = df[data_column].apply(lambda x: x.replace('In Case of Emergency Dial 911 or nearest Emergency Room National Suicide Prevention Life Line 1-800-273-TALK or 1-800-273-8255 http://www.suicidepreventionlifeline.org', ' '))
#     df[f'{data_column}'] = df[data_column].apply(lambda x: x.replace('Reduce potential for lethal means Family and Friends should assist in securing or removing from the house or car all means of committing suicide impulsively', ' '))
#     df[f'{data_column}'] = df[data_column].apply(lambda x: x.replace('National Alliance on Mental Illness 1-801-323-9900 or 1-877-230-6264 http://www.nami.org Substance Abuse and Mental Health Services Administration 1-877-726-4727 http://www.samhsa.gov/suicide-prevention', ' '))
#
#     return df


def remove_tags(html):

    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()

    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)



def remove_this(x):
    #Function Defintion: This function removes unncessary punctuation from the string
    x = x.strip()
    if x == ',' or x == ')' or x== '(' or x == ';' or x == '/' or x == '[' or x == ']':
        return False
    return True


def preprocess_note(this_note, suicide_asssments):
    #Function Definition: This function preprocesses the note to get rid of specific punctuations, the year, dates, and suicide assessments
    
    this_note = this_note.replace('-', ' ')
    this_note = this_note.replace('/', ' ')
    
    this_note = re.sub('[1-3][0-9]{3}', 'xYEARx', this_note)
    this_note = re.sub('((0?[13578]|10|12)(-|\/)(([1-9])|(0[1-9])|([12])([0-9]?)|(3[01]?))(-|\/)((19)([2-9])(\d{1})|(20)([01])(\d{1})|([8901])(\d{1}))|(0?[2469]|11)(-|\/)(([1-9])|(0[1-9])|([12])([0-9]?)|(3[0]?))(-|\/)((19)([2-9])(\d{1})|(20)([01])(\d{1})|([8901])(\d{1})))$', 'xDATEx', this_note)
    this_note = re.sub('(0?[1-9]|[12][0-9]|3[01])/[12][0-9]|3[01]', 'xDATEx', this_note)
    
    for s in suicide_asssments:
        this_note = this_note.replace(s, '')

    return this_note

def substring_period(sub_array, values, idx):
    # Function Defintion: This function returns the substring inputted into the negex algorithm when the phrase sends with a period.
    #Steps:
    #1) Remove extraneous characters
    #2) Checks to make sure the string is greater than 0
    #3) Check to see whether the substring is long in length.
    #4) If so, we find the index of the SI term and select a short substring surrounding that term
    #5) Returns the substring
    
    sub_array = [sa for sa in sub_array if remove_this(sa)]
    
    if len(sub_array) > 0:
    
        suicidal_term_index = sub_array.index(values[idx])
    
    # if we have a very long sentence
        if len(sub_array) > 20:
            dist = suicidal_term_index - 10
            if dist < 0:
                max_dist = suicidal_term_index - 0
            else:
                max_dist = 10
            sub_array = sub_array[suicidal_term_index -max_dist: suicidal_term_index + 2]
            suicidal_term_index = sub_array.index(values[idx])

        substring = ' '.join(sub_array)

        return substring
    else:
        return ''

def SI_distance(sub_array, question, values, idx):
    #Function Definition: This function returns the distance between the suicidal ideation term and the sentence break to determine if the substring is natural or structured.
    #Steps:
    #1a) If the sentence break is a question mark, find the index of the question mark
    #1b) If the sentence break is a colon, find the index of the colon
    #2) Find the index of the suicidal ideation term
    #3) Return the distance between the break index and the suicidal ideation term
    
    if question == True:
        break_index = sub_array.index('?',1)
    else:
        break_index = sub_array.index(':',1)

    suicidal_term_index = sub_array.index(values[idx])
    distance = break_index - suicidal_term_index

    return distance


def natural_tagger(substring, SI_phrases, irules):
    #Function Definition: This function returns the negation flag for substrings written in natural language form.
    tagger = negTagger(sentence = substring, phrases = SI_phrases ,rules = irules, negP = False)
    if tagger.getNegationFlag() == "affirmed":
        return False
    else:
        return True

def structured_tagger(substring, SI_phrases, irules2):
    #Function Definition: This function returns the negation flag for substrings written in structured language form.
    tagger = negTagger(sentence = substring, phrases = SI_phrases ,rules = irules2, negP = False)
    if tagger.getNegationFlag() == "affirmed":
        return False
    else:
        return True


def find_all_indexes(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1

def preprocess_note2(this_text):

    our_punct = '!"#$%&\'()*+,-/;<=>@[\\]^_`{|}~·'
    translator = str.maketrans(our_punct, ' '*len(our_punct))
    this_text = this_text.translate(translator)

    this_text = re.sub('[1-3][0-9]{3}', 'xYEARx', this_text)

    this_text = re.sub('((0?[13578]|10|12)(-|\/)(([1-9])|(0[1-9])|([12])([0-9]?)|(3[01]?))(-|\/)((19)([2-9])(\d{1})|(20)([01])(\d{1})|([8901])(\d{1}))|(0?[2469]|11)(-|\/)(([1-9])|(0[1-9])|([12])([0-9]?)|(3[0]?))(-|\/)((19)([2-9])(\d{1})|(20)([01])(\d{1})|([8901])(\d{1})))$', 'xDATEx', this_text)
    this_text = re.sub('(0?[1-9]|[12][0-9]|3[01])/[12][0-9]|3[01]', 'xDATEx', this_text)

    return this_text

# from bs4 import BeautifulSoup, NavigableString
#
# def html_to_text(html):
#     "Creates a formatted text email message as a string from a rendered html template (page)"
#     soup = BeautifulSoup(html, 'html.parser')
#     # Ignore anything in head
#     text = soup
#     for element in text:
#         # We use type and not isinstance since comments, cdata, etc are subclasses that we don't want
#         if type(element) == NavigableString:
#             # We use the assumption that other tags can't be inside a script or style
#             if element.parent.name in ('script', 'style'):
#                 continue
#
#             # remove any multiple and leading/trailing whitespace
#             string = ' '.join(element.string.split())
#             if string:
#                 if element.parent.name == 'a':
#                     a_tag = element.parent
#                     # replace link text with the link
#                     string = a_tag['href']
#                     # concatenate with any non-empty immediately previous string
#                     if (    type(a_tag.previous_sibling) == NavigableString and
#                             a_tag.previous_sibling.string.strip() ):
#                         text[-1] = text[-1] + ' ' + string
#                         continue
#                 elif element.previous_sibling and element.previous_sibling.name == 'a':
#                     text[-1] = text[-1] + ' ' + string
#                     continue
#                 elif element.parent.name == 'p':
#                     # Add extra paragraph formatting newline
#                     string = '\n' + string
#                 text += [string]
#     doc = '\n'.join(text)
#     return doc