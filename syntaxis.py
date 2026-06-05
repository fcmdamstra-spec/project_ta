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
import sys 

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

def determine_syntax_story(doc):
    """ 
    Predicts story or nonstory based on: Rules obtained from syntactic feature analysis:
    - POS TAGS: stories use more (12,2%) PRON tags than nonstories (10.8%).  PROPN only appears in non-story top 10 (3.4%).
    - DEPENDENCIES: not very big differences
    - NOUN CHUNKS: avg of noun chunk differs ~ 34%, stories: 76, nonstories: 57
    - SENTENCE LENGTH: not useful, almost the same.

    Rule 1 (POS): if n of PRON% > 11.5%, predict: story
    Rule 2 (POS): if PROPN% > 3%, predict: non-story
    Rule 3 (NOUN CHUNKS): if avg noun chunks > 65, predict: story

    Voting: majority of 3 decides label.
    """

    _, pos_pct = count_pos(doc)
    chunk_count, _, _ = analyze_noun_chunk(doc)

    votes_story    = 0
    votes_nonstory = 0

    # Rule: PRON frequency
    if pos_pct.get('PRON', 0) > 11.5:
        votes_story += 1
    else:
        votes_nonstory += 1

    # Rule: noun chunk count
    if chunk_count > 65:
        votes_story += 1
    else:
        votes_nonstory += 1

    # Rule: PROPN frequency
    if pos_pct.get('PROPN', 0) > 3.0:
        votes_nonstory += 1
    else:
        votes_story += 1

    return 'story' if votes_story > votes_nonstory else 'non-story'

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
        chunk_count, _, _ = analyze_noun_chunk(doc)
        if (pos_pct.get('PRON', 0) > 11.5) == (label == 'story'): 
            rule1_correct += 1
        if (pos_pct.get('PROPN', 0) <= 3.0) == (label == 'story'): 
            rule2_correct += 1
        if (chunk_count > 65) == (label == 'story'): 
            rule3_correct += 1
        if determine_syntax_story(doc) == label: 
            voting_correct += 1

    f = open("syntax_patterns.txt", "w", encoding="utf-8")
    sys.stdout = f

    # print results / feature
    print(" OBSERVATION FROM DEVELOPMENT DATA:\n POS TAGS\n")
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
    

    print(f"""
    SYNTAX PATTERNS

    1. POS:
    
    - Pronoun (PRON) frequency
    Observation: stories averaged {story_pos_pct['PRON']/stories:.1f}% PRON vs {nonstory_pos_pct['PRON']/nonstories:.1f}% for non-stories.
    Method:      spaCy token.pos_
    Rule:        If PRON% > 11.5% predict 'story'
    Accuracy:    {rule1_correct/total*100:.1f}%
    Works when:  text is a first-person narrative for example using "I, me, we".
    Fails when:  a non-story is conversational and also uses many pronouns.

    - Proper noun (PROPN) frequency
    Observation 2: non-stories averaged {nonstory_pos_pct['PROPN']/nonstories:.1f}% PROPN vs {story_pos_pct['PROPN']/stories:.1f}% for stories.
    Method:      spaCy token.pos_
    Rule:        If PROPN% > 3.0% predict 'non-story'
    Accuracy:    {rule2_correct/total*100:.1f}%
    Works when:  text has many named entities (names, people, things)
    Fails when:  a story mentions real names or places frequently.

    2. DEPENDENCIES - not used!
    Observation: see dependency distribution printed above.
    Method:      spaCy token.dep_, doc.noun_chunks
    Rule:        not included because distributions too similar between classes.
    Accuracy:    N/A
    Works when:  N/A
    Fails when:  N/A

    3. Noun chunk count - used as rule
    Observation: stories averaged {sum(story_chunks)/len(story_chunks):.1f} chunks vs {sum(nonstory_chunks)/len(nonstory_chunks):.1f} for non-stories.
    Method:      spaCy doc.noun_chunks
    Rule:        If the noun chunk count > 65 predict 'story'
    Accuracy:    {rule3_correct/total*100:.1f}%
    Works when:  longer narrative texts have more noun phrases.
    Fails when:  a long non-story has equally many noun chunks.

    4. Sentence length - not used!
    Observation: stories averaged {sum(story_sent_lengths)/len(story_sent_lengths):.1f} tokens/sent vs {sum(nonstory_sent_lengths)/len(nonstory_sent_lengths):.1f} for non-stories.
    Method:      spaCy doc.sents
    Rule:        not included, difference too small.
    Accuracy:    N/A
    Works when:  N/A
    Fails when:  N/A

    Combined voting accuracy (majority of 3 rules): {voting_correct/total*100:.1f}%
    """)

    f.close()
    sys.stdout = sys.__stdout__  


if __name__ == "__main__":
    main()


