"""
Part 1 - Syntactical Analysis

Created by: Fleur, S4109287

Classifies Reddit posts as 'story' or 'non-story' using three syntactic features:
  1. Part of Speech (POS) - tagging (frequency/ratio and n-grams): Spacy token.pos_ 
  2. Syntactic Parsing (dependency): Spacy token.dep_, token.pos_
  3. Sentence length

Order: 
  Part 2, Feature extraction functions
  Part 3, Development data analysis (main)
  Part 4, Classification rules, derived from 3 (syntax_breakdown)
  
"""

import spacy
from collections import Counter
import pandas as pd
import sys 

nlp = spacy.load('en_core_web_sm')
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"


''' 
Part 2: feature extraction functions (POS, Dependency Parsing & Sentence length) to determine rules
'''

### 1: POS ANALYSIS: FREQUENCY AND N_GRAMS
def count_pos(doc):
    """ Returns raw POS-tag counts and percentages """

    pos_count = Counter()
    for token in doc:
        pos_count[token.pos_] += 1
    
    total = len(doc)
    pos_pct = {tag: (count/total) * 100 for tag, count in pos_count.items()}
    return pos_count, pos_pct

def pos_ngrams(doc):
    """ Returns POS tag bigrams and trigrams (sequence pattern - bigrams, trigrams)"""
    bigrams = Counter()
    trigrams = Counter()

    tags = [token.pos_ for token in doc]

    for i in range(len(tags)-1):
        bigrams[(tags[i], tags[i+1])] += 1
    for i in range(len(tags)-2):
        trigrams[(tags[i], tags[i+1], tags[i+2])] += 1

    return bigrams, trigrams


### 2: DEPENDENCY PARSING
def dependency_pos_analysis(doc):
    """ Returns Descriptive analysis of dependency + POS combinations
     - dep_counts: raw dependency frequencies
     - dep_pos: (dependency, POS) combinations, dep_pos_pct: frequency percentage
    """

    dep_counts = Counter()
    dep_pos = Counter()

    for token in doc:
        dep_counts[token.dep_] += 1  # raw dependency label frequency
        dep_pos[(token.dep_, token.pos_)] += 1 #dependency + POS of token 

    total= len(doc)
    dep_pos_pct = {pair: (count / total) * 100 for pair, count in dep_pos.items()} #frequency

    return dep_counts, dep_pos, dep_pos_pct


### 3: SENTENCE LENGTH
def avg_sent_length(doc):
    """ Returns average sentence length for a text"""
    sentence_length = []
    for sent in doc.sents:
        sentence_length.append(len(sent))
   
    return sum(sentence_length) / len(sentence_length) if sentence_length else 0



'''
Part 4: classification rules derived form Part 3
'''

def syntax_breakdown(doc):
    """ Returns each syntactic sub-rule decision plus the combined syntax label.

    Rules obtained from syntactic feature analysis:
    - POS TAGS: stories use more (12,2%) PRON tags than nonstories (10.8%).  PROPN in non-story (3.4%), vs story: 2,7%.
    - DEPENDENCY  PARSING: DEP-POS: ('advmod', 'ADV'): stories (5.71%), non-story: (4.88%).
    - SENTENCE LENGTH: not useful, almost the same.

    Rule 1 (POS): if n of PRON% > 11.5%, predict: story
    Rule 2 (POS): if PROPN% > 3%, predict: non-story
    Rule 3 (DEPENDENCY): if advmod+ADV% > 5.3% predict story


    Voting: 2 or more 'story' votes (majority of 3) makes the combined label 'story'.
    Single source of truth used by both determine_syntax_story and the logging in main.py.
    """

    _, pos_pct = count_pos(doc)
    _, _, dep_pos_pct = dependency_pos_analysis(doc)
    
    decisions = {
        'syntax_pron': 'story' if pos_pct.get('PRON', 0) > 11.5 else 'non-story',
        'syntax_propn': 'non-story' if pos_pct.get('PROPN', 0) > 3.0 else 'story',
        'syntax_dep': 'story' if dep_pos_pct.get(('advmod', 'ADV'), 0) > 5.3  else 'non-story',
    }
    votes = list(decisions.values()).count('story')
    decisions['syntax'] = 'story' if votes >= 2 else 'non-story'
    return decisions


def determine_syntax_story(doc):
    """ Predicts story or non-story by majority vote over the syntactic sub-rules. """

    return syntax_breakdown(doc)['syntax']



'''
Part 3: run feature tests on dev dataset to identify patterns
'''

def main():   
    """ Runs feature extraction functions from part 2 on dev dataset returns POS, dependency and sentence length feature results """
    choice = input("Run on (d)ev or (t)est dataset? [d/t]: ").strip().lower()
    csv_file = DEV_CSV if choice == 'd' else TEST_CSV
    df = pd.read_csv(csv_file)
    
    story_pos = Counter()
    nonstory_pos = Counter()
    story_pos_pct = Counter()
    nonstory_pos_pct = Counter()
    story_bigrams = Counter()
    nonstory_bigrams = Counter()
    story_trigrams = Counter()
    nonstory_trigrams = Counter()
    story_dep_counts = Counter()
    nonstory_dep_counts = Counter()
    story_dep_pos = Counter()
    nonstory_dep_pos = Counter()
    story_dep_pos_pct = Counter()
    nonstory_dep_pos_pct = Counter()
    story_sent_lengths = []
    nonstory_sent_lengths = []

    stories = 0
    nonstories = 0

    for _, row in df.iterrows():
        text  = row['content']
        label = row['label']
        doc = nlp(text)

        pos, pos_pct = count_pos(doc)
        bigrams, trigrams = pos_ngrams(doc) 
        dep_counts, dep_pos, dep_pos_pct = dependency_pos_analysis(doc)
        length = avg_sent_length(doc)
        

        # update counts of features for stories:
        if label == "story":

            stories += 1
            story_pos.update(pos)
            story_pos_pct.update(pos_pct)
            story_bigrams.update(bigrams)
            story_trigrams.update(trigrams)
            story_dep_counts.update(dep_counts)
            story_dep_pos.update(dep_pos)
            story_dep_pos_pct.update(dep_pos_pct)
            story_sent_lengths.append(length)
            

        # update counts of features for nonstories:
        else:

            nonstories += 1
            nonstory_pos.update(pos)
            nonstory_pos_pct.update(pos_pct)
            nonstory_bigrams.update(bigrams)
            nonstory_trigrams.update(trigrams)
            nonstory_sent_lengths.append(length)  
            nonstory_dep_counts.update(dep_counts)
            nonstory_dep_pos.update(dep_pos)
            nonstory_dep_pos_pct.update(dep_pos_pct)
            

    #Accuracy: 
    rule1_correct = 0
    rule2_correct = 0
    rule3_correct = 0
    voting_correct = 0
    total = len(df)

    for _, row in df.iterrows():
        doc = nlp(row['content'])
        label = row['label']
        _, pos_pct = count_pos(doc)
        _, _, dep_pos_pct = dependency_pos_analysis(doc)
        

        if (pos_pct.get('PRON', 0) > 11.5) == (label == 'story'): 
            rule1_correct += 1
        if (pos_pct.get('PROPN', 0) <= 3.0) == (label == 'story'): 
            rule2_correct += 1
        if (dep_pos_pct.get(('advmod', 'ADV'), 0) > 5.3) == (label == 'story'):
            rule3_correct += 1

        if determine_syntax_story(doc) == label: 
            voting_correct += 1

    f = open("syntax_patterns.txt", "w", encoding="utf-8")
    sys.stdout = f

    # print results / feature
    print(" OBSERVATION FROM DEVELOPMENT DATA:\n POS TAGS\n")
    print("Story POS tags: \n") 
    for tag, count in story_pos.most_common(15):
        avg_pct = story_pos_pct[tag] / stories
        print(f"{tag}:{count} ({avg_pct:.1f}%)")
    
    print("Non-story POS tags:\n")
    for tag, count in nonstory_pos.most_common(15):
        avg_pct = nonstory_pos_pct[tag] / nonstories
        print(f"{tag}: {count} ({avg_pct:.1f}%)")

    print("\nBIGRAMS\n")
    print("POS bigrams in Stories:")
    total_story_bigrams = sum(story_bigrams.values())
    for bigram, count in story_bigrams.most_common(10):
        pct = count / total_story_bigrams * 100
        print(f"{' '.join(bigram)}: {count} ({pct:.1f}%)")

    print("\nPOS bigrams in Non-Stories:")
    total_nonstory_bigrams = sum(nonstory_bigrams.values())
    for bigram, count in nonstory_bigrams.most_common(10):
        pct = count / total_nonstory_bigrams * 100
        print(f"{' '.join(bigram)}: {count} ({pct:.1f}%)")

    print("\nPOS trigrams in Stories:")
    total_story_trigrams = sum(story_trigrams.values())
    for trigram, count in story_trigrams.most_common(10):
        pct = count / total_story_trigrams * 100
        print(f"{' '.join(trigram)}: {count} ({pct:.1f}%)")

    print("\nPOS trigrams in Non-Stories:")
    total_nonstory_trigrams = sum(nonstory_trigrams.values())
    for trigram, count in nonstory_trigrams.most_common(10):
        pct = count / total_nonstory_trigrams * 100
        print(f"{' '.join(trigram)}: {count} ({pct:.1f}%)")
    
    print("\nDEPENDENCY PARSING\n")
    print("\nDEP + POS Story")
    for pair, c in story_dep_pos.most_common(10):
        avg_pct = story_dep_pos_pct[pair] / stories
        print(f"{pair}: {c} ({avg_pct:.2f}%)")

    print("\nDEP + POS Non-story")
    for pair, c in nonstory_dep_pos.most_common(10):
        avg_pct = nonstory_dep_pos_pct[pair] / nonstories
        print(f"{pair}: {c} ({avg_pct:.2f}%)")


    print("\nSENTENCE LENGTH\n")
    print(f"Average sentence length in Stories: "
        f"{sum(story_sent_lengths)/len(story_sent_lengths)}\n")

    print(
        f"Average sentence length in Non-Stories: "
        f"{sum(nonstory_sent_lengths)/len(nonstory_sent_lengths)}")
    

    print(f"""
    SYNTAX PATTERNS

    POS:
    
    Rule 1 - Pronoun (PRON) frequency
    Observation: stories averaged {story_pos_pct['PRON']/stories:.1f}% PRON vs {nonstory_pos_pct['PRON']/nonstories:.1f}% for non-stories.
    Method:      spaCy token.pos_
    Rule:        If PRON% > 11.5% predict 'story'
    Accuracy:    {rule1_correct/total*100:.1f}%
    Works when:  story is a first-person narrative for example using "I, me, we"
    Fails when:  a non-story is conversational and can also use many pronouns

    Rule 2 - Proper noun (PROPN) frequency
    Observation 2: non-stories averaged {nonstory_pos_pct['PROPN']/nonstories:.1f}% PROPN vs {story_pos_pct['PROPN']/stories:.1f}% for stories.
    Method:      spaCy token.pos_
    Rule:        If PROPN% > 3.0% predict 'non-story'
    Accuracy:    {rule2_correct/total*100:.1f}%
    Works when:  non-stpry is descriptive and has many named entities (names, people, things)
    Fails when:  overall weak pattern, because stories use real names or places frequently too

    POS - Tag distributions, N-grams (Bigrams - Trigrams) - Not used. no pattern shows a large enough difference to make a reliable rule

    DEPENDENCY: 
   
    Rule 3 - Adverbial modifier (advmod+ADV) frequency
    Observation: stories averaged {story_dep_pos_pct[('advmod', 'ADV')]/stories:.2f}% advmod+ADV vs {nonstory_dep_pos_pct[('advmod', 'ADV')]/nonstories:.2f}% for non-stories.
    Method:      spaCy token.dep_, token.pos_
    Rule:        If advmod+ADV% > 5.3% predict 'story'
    Accuracy:    {rule3_correct/total*100:.1f}%
    Works when:  stories use narrative words like "then" and descriptive words.
    Fails when:  nonstories could contain adverbs for descriptions in a non-narrative way too

    SENTENCE LENGTH:  - not used! Differences were too small

    Sentence length
    Observation: stories averaged {sum(story_sent_lengths)/len(story_sent_lengths):.1f} tokens/sent vs {sum(nonstory_sent_lengths)/len(nonstory_sent_lengths):.1f} for non-stories.
    Method:      spaCy doc.sents
 

    Combined voting accuracy (majority of 3 rules): {voting_correct/total*100:.1f}%
    """)

    f.close()
    sys.stdout = sys.__stdout__  


if __name__ == "__main__":
    main()