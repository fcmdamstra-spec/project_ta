"""
Part 1 - Syntactical Analysis

Created by: Fleur, S4109287

Classifies Reddit posts as 'story' or 'non-story' using four syntactic features:
  1. Part of Speech (POS) - tagging: Spacy token.pos_ 
  2. Syntactic Parsing (dependency + noun chunks): Spacy token.dep_, Spacy doc.noun_chunks
  3. Sentence length

Order: 
  Part 2, Feature extraction functions
  Part 3, Development data analysis (main)
  Part 4, Classification rules, derived from 3 (determine_syntax_story)
  
"""

import spacy
from collections import Counter
import pandas as pd

nlp = spacy.load('en_core_web_sm')
DEV_CSV = "dev.csv"

''' 
Part 2: feature extraction functions (POS, Parsing, Chunks & Sentence length) to determine rules
'''

def count_pos(doc):
    """ Returns raw POS-tag counts and percentages """

    pos_count = Counter()
    for token in doc:
        pos_count[token.pos_] += 1
    
    total = len(doc)
    pos_pct = {tag: (count/total) * 100 for tag, count in pos_count.items()}
    return pos_count, pos_pct


def analyze_noun_chunk(doc):
    """ Returns , for each noun chunk:
    - noun chunk text
    - root word
    - dependency role
    - head it connects to """
    
    chunk_deps = Counter()
    chunk_texts = Counter()
    for chunk in doc.noun_chunks:
        chunk_deps[chunk.root.dep_] += 1
        if len(chunk) > 1:
            chunk_texts[chunk.text.lower()] += 1
    
    return len(list(doc.noun_chunks)), chunk_deps, chunk_texts


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
    """
    Returns each syntactic sub-rule decision plus the combined syntax label.

    Rules obtained from syntactic feature analysis:
    - POS TAGS: stories use more (12,2%) PRON tags than nonstories (10.8%).  PROPN only appears in non-story top 10 (3.4%).
    - DEPENDENCIES: not very big differences
    - NOUN CHUNKS: avg of noun chunk differs ~ 34%, stories: 76, nonstories: 57
    - SENTENCE LENGTH: not useful, almost the same.

    Rule (POS): if n of PRON% > 11.5%, predict: story
    Rule (POS): if PROPN% > 3%, predict: non-story
    Rule (NOUN CHUNKS): if avg noun chunks > 65, predict: story

    Voting: 2 or more 'story' votes (majority of 3) makes the combined label 'story'.
    Single source of truth used by both determine_syntax_story and the logging in main.py.
    """

    _, pos_pct = count_pos(doc)
    chunk_count, _, _ = analyze_noun_chunk(doc)

    decisions = {
        'syntax_pron': 'story' if pos_pct.get('PRON', 0) > 11.5 else 'non-story',
        'syntax_noun_chunk': 'story' if chunk_count > 65 else 'non-story',
        'syntax_propn': 'non-story' if pos_pct.get('PROPN', 0) > 3.0 else 'story',
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
    df = pd.read_csv(DEV_CSV)
    
    story_pos = Counter()
    nonstory_pos = Counter()
    story_pos_pct = Counter()
    nonstory_pos_pct = Counter()

    story_chunk_deps    = Counter()
    nonstory_chunk_deps = Counter()
    story_chunk_texts    = Counter()
    nonstory_chunk_texts = Counter()
    story_chunks    = []
    nonstory_chunks = []

    story_sent_lengths = []
    nonstory_sent_lengths = []

    stories = 0
    nonstories = 0

    for _, row in df.iterrows():
        text  = row['content']
        label = row['label']
        doc = nlp(text)

        pos, pos_pct = count_pos(doc)
        chunk_count, chunk_deps, chunk_texts = analyze_noun_chunk(doc)
        length = avg_sent_length(doc)

        # update counts of features for stories:
        if label == "story":

            stories += 1
            story_pos.update(pos)
            story_chunks.append(chunk_count)
            story_chunk_texts.update(chunk_texts)
            story_chunk_deps.update(chunk_deps)
            story_sent_lengths.append(length)
            story_pos_pct.update(pos_pct)

        # update counts of features for nonstories:
        else:

            nonstories += 1
            nonstory_pos.update(pos)
            nonstory_chunks.append(chunk_count)
            nonstory_chunk_deps.update(chunk_deps)
            nonstory_chunk_texts.update(chunk_texts)
            nonstory_sent_lengths.append(length)  
            nonstory_pos_pct.update(pos_pct)



    # print results / feature
    print("POS TAGS\n")
    print("Story: \n") 
    for tag, count in story_pos.most_common(10):
        avg_pct = story_pos_pct[tag] / stories
        print(f"{tag}: {count} ({avg_pct:.1f}%)")
    
    print("Non-story:\n")
    for tag, count in nonstory_pos.most_common(10):
        avg_pct = nonstory_pos_pct[tag] / nonstories
        print(f"  {tag}: {count} ({avg_pct:.1f}%)")


    print("\nNOUN CHUNKS\n")
    print(f"Average noun chunks in Stories: "
          f"{sum(story_chunks)/len(story_chunks)}\n")
    print(f"Average noun chunks in Non- Stories: "
          f"{sum(nonstory_chunks)/len(nonstory_chunks)}\n")
    
    print("Top 10 multiword noun chunks in Stories:\n")
    for chunk, count in story_chunk_texts.most_common(10):
        print(f"{chunk}: {count}")
    print("Top 10 multiword noun chunks in Non-Stories:\n")
    for chunk, count in nonstory_chunk_texts.most_common(10):
        print(f"{chunk}: {count}")

    print("\nNounchunk dependency roles Stories:")
    for dep, count in story_chunk_deps.most_common():
        pct = (count / sum(story_chunk_deps.values())) * 100
        print(f"{dep}: {count} ({pct:.1f}%)")

    print("\nNounchunk dependency roles in Non-Stories:")
    for dep, count in nonstory_chunk_deps.most_common():
        pct = (count / sum(nonstory_chunk_deps.values())) * 100
        print(f"{dep}: {count} ({pct:.1f}%)")


    print("\nSENTENCE LENGTH\n")
    print(f"Average sentence length in Stories: "
        f"{sum(story_sent_lengths)/len(story_sent_lengths)}\n")

    print(
        f"Average sentence length in Non-Stories: "
        f"{sum(nonstory_sent_lengths)/len(nonstory_sent_lengths)}")
if __name__ == "__main__":
    main()
