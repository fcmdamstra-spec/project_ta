"""
Part 2 - Semantic Analysis

Created by: Cas, S5414865

Classifies Reddit posts as 'story' or 'non-story' using four semantic features:
  1. Coreference resolution   - using corefee to count coreference chains
  2. Named-entity recognition - spaCy doc.ents
  3. WordNet noun count       - NLTK WordNet
  4. WSD ambiguity proportion - proportion of content words with multiple synsets

Each feature produces one rule. The four rules votes, 2 or more votes for story and the text is determined to be a story.
"""

from nltk.corpus import wordnet
import coreferee
import spacy
import pandas as pd


nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('coreferee')

'''
Part 1: determining story or non-story per semantic method 
'''

def coref_story_decision(doc):
    """
    Count coreference chains using coreferee.
    
    Each chain groups all mentions of the same entity.
    More chains → more entities being tracked → more likely a story.

    Based on analysis of the data stories averaged 6.2 coreference chain length vs 4.3 for non stories
    Using this if the given doc contains 5 or more coreference chains it is predicted to be a story
    """
    if len(doc._.coref_chains) >= 5:
        return 'story'
    else:
        return 'non-story'

    
def ner_story_decision(doc):
    """
    Count all named entities using spaCy's built-in NER.

    Based on analysis of the data stories averaged 9.9 named entities vs 8.4 for non-stories
    Using this when a doc has 9 or more entities it is predicted to be a story
    """
    if len(doc.ents) >=9:
        return 'story'
    else:
        return 'non-story'

    
def count_wordnet_nouns(doc):
    """
    Count nouns that exist as WordNet entries (have at least one synset).

    Stories tend to use more concrete nouns (people, objects, places) than
    non-stories, which may use more abstract or technical vocabulary.
    """
    wn_nouns = [
        token for token in doc
        if token.pos_ == 'NOUN' and wordnet.synsets(token.lemma_)
    ]
    return len(wn_nouns)


def wordnet_noun_story_decision(doc):
    '''
    Stories averaged 45.1 WordNet nouns vs 35.5 for non-stories
    based on this when a string contains 40 or more wordnet noun counts it is predicted to be a story
    '''
    if count_wordnet_nouns(doc) >= 40:
        return 'story'
    else:
        return 'non-story'

    
def count_ambiguous_features(doc):
    """
    Proportion of content words (NOUN / VERB / ADJ, non-stop) that are
    'ambiguous', meaning WordNet lists more than one synset for them.
    """
    ambiguous = 0

    for token in doc:
        if not token.is_stop and not token.is_punct and token.pos_ in ('NOUN', 'VERB', 'ADJ') and len(wordnet.synsets(token.lemma_)) > 1:
            ambiguous += 1
    return ambiguous


def ambiguous_story_decision(doc):
    if count_ambiguous_features(doc) >= 72:
        return 'story'
    else:
        return 'non-story'

'''
Part 2: Determining story or non-story by combing all different methods
'''
    
def semantics_breakdown(doc):
    '''
    Returns each semantic sub-rule decision plus the combined semantics label.
    The 4 methods vote; 2 or more 'story' votes makes the combined label 'story'.
    This is the single source of truth used by both determine_semantics_story and
    the logging in main.py.
    '''
    decisions = {
        'coref': coref_story_decision(doc),
        'ner': ner_story_decision(doc),
        'wordnet_noun': wordnet_noun_story_decision(doc),
        'ambiguous': ambiguous_story_decision(doc),
    }
    votes = list(decisions.values()).count('story')
    decisions['semantics'] = 'story' if votes >= 2 else 'non-story'
    return decisions


def determine_semantics_story(doc):
    '''
    determine_semantics_story decides wheter a given stext tory (as spacy doc) is a story or not
    It does so by letting each semantic method vote, since there are 4 methods the threshold for story is 2 or more
    '''
    return semantics_breakdown(doc)['semantics']
    

    

DEV_CSV = "dev.csv"
def main():
    '''
    The main function in this script was used during testing and calibrating the other functions in semantics.py.
    The eventual programm that determines story or non-story for the assignment can be found in main.py
    '''
    total = 0
    stories = 0
    correct = 0

    coref_correct = 0
    non_story_coref= []
    story_coref = []

    ner_correct = 0
    non_story_ner= []
    story_ner = []

    wnnoun_correct = 0
    non_story_wnnoun = []
    story_wnnoun = []

    wsd_correct = 0
    non_story_wsd = []
    story_wsd = []


    df = pd.read_csv(DEV_CSV)
    for _, row in df.iterrows():
        text  = row['content']
        label = row['label']
        doc = nlp(text)
        # determine coreference story decision and count correct
        if coref_story_decision(doc) == label:
            coref_correct += 1

        # determine NER story decision and count correct
        if ner_story_decision(doc) == label:
            ner_correct += 1

        # determine wordnet noun decision and count correct
        if wordnet_noun_story_decision(doc) == label:
            wnnoun_correct += 1

        #determine wsd decision and count correct
        if ambiguous_story_decision(doc) == label:
            wsd_correct += 1

        if determine_semantics_story(doc) == label:
            correct += 1

        # count features per label to determine averages
        if label == 'story':
            stories += 1
            story_coref.append(len(doc._.coref_chains))
            story_ner.append(len(doc.ents))
            story_wnnoun.append(count_wordnet_nouns(doc))
            story_wsd.append(count_ambiguous_features(doc))
        else:
            non_story_coref.append(len(doc._.coref_chains))
            non_story_ner.append(len(doc.ents))
            non_story_wnnoun.append(count_wordnet_nouns(doc))
            non_story_wsd.append(count_ambiguous_features(doc))

        total += 1
    non_stories = total - stories

    print (f'percentage coref correct: {round(coref_correct/total * 100, 1)}')
    print(f'average coreference chain length for story: {sum(story_coref)/stories}')
    print(f'average coreference chain length for non-story: {sum(non_story_coref)/non_stories}')
    print(f'percentage NER correct: {round(ner_correct/total * 100, 1)}')
    print(f'average ner count for story: {sum(story_ner)/stories}')
    print(f'average ner count for non-story: {sum(non_story_ner)/non_stories}')
    print(f'percentage wordnet nouns correct: {round(wnnoun_correct/total *100, 1)}')
    print(f'average wornet nouns count for story: {sum(story_wnnoun)/stories}')
    print(f'average wordnet nouns count for non-story: {sum(non_story_wnnoun)/non_stories}')
    print(f'percentage wsd correct: {round(wsd_correct/total *100, 1)}')
    print(f'average wsd count for story: {sum(story_wsd)/stories}')
    print(f'average wsd count for non-story: {sum(non_story_wsd)/non_stories}')
    print(f'percentage all semantic rules correct: {round(correct/total *100, 1)}')


if __name__ == "__main__":
    main()