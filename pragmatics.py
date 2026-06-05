"""
Part 3 - Pragmatics Analysis
Created by Frederique, s5042968

Classifies Reddit posts as 'story' or 'non-story' using three pragmatic features
based on sentence-level sentiment analysis (VADER via asent):

1. Emotional range = difference between the most positive and negative sentence
2. Sentiment shift = number of times tone changes direction across sentences
3. Sentence count = stories tend to be longer and use more sentences

Each feature produces one rule. The three rules vote: 2 or more votes -> predicted story

Observation from dev data: 
    Stories averaged an emotional range of 1.14 vs 1.04 for non-stories
    Stories averaged 5.60 sentiment shifts vs 4.46 for non-stories
    Stories averaged 14.26 sentences vs 10.62 for non-stories
"""

import spacy
import asent 
import pandas as pd

nlp_sentiment = spacy.blank('en')
nlp_sentiment.add_pipe('sentencizer')
nlp_sentiment.add_pipe('asent_en_v1')


#Extraction helpers

def get_sentence_sentiments(text: str) -> list[float]:
    """
    Return a list of compound sentiment scores, one per sentence.
    The compound score ranges from -1 to +1. A score > 0.05 is considered positive, 
    <-0.05 negative and in between neutral. 
    = standard VADER thresholds. 
    """

    doc = nlp_sentiment(text)
    return[sent._polarity.compound for sent in doc.sents]

def emotional_range(sent_scores: list[float]) -> float:
    """
    The difference between the highest and lowest sentence-level compound scores.
    A larger range indicates more emotional variety: the text swings between
    positive and negative, which is one of the characteristic of narrative stories.
    """

    if len(sent_scores) < 2:
        return 0.0
    return max(sent_scores) - min(sent_scores)

def count_sentiment_shifts(sent_scores: list[float]) -> int:
    """
    Count how many times the sentiment direction changes between consecutive sentences.
    A shift is counted when a sentence's polarity category (positive / non-positive)
    differs from the previous sentence. Stories have more variation between positive and negative moments than 
    informational texts.
    """

    if len(sent_scores) < 2:
        return 0
    shifts = 0
    for i in range(1, len(sent_scores)):
        prev_positive = sent_scores[i - 1] > 0.05
        curr_positive = sent_scores[i] > 0.05
        if prev_positive != curr_positive:
            shifts += 1
    return shifts

#Individual decision rules

def emotional_range_decision(text: str) -> str:
    """
    Rule 1: emotional range
    Observation: Stories averaged an emotional range of 1.14 vs 1.04 for non-stor-*ies
    on the development set.
    Rule: if emotional_range >= 0.9, predict 'story'.
    Dev accuracy: 55.0%
 
    Works when: the text contains clear positive and negative moments.
    Fails when: a story has a consistent tone throughout, or a non-story uses
    strong positive/negative language (e.g. a rant or a glowing recommendation).
    """

    scores = get_sentence_sentiments(text)
    if emotional_range(scores) >= 0.9:
        return 'story'
    return 'non-story'

def sentiment_shift_decision(text: str) -> str:
    """
    Rule 2: Sentiment shifts.
 
    Observation: Stories averaged 5.60 sentiment shifts vs 4.46 for non-stories
    in the observation data.
    Rule: if number of shifts >= 5, predict 'story'.
    Dev accuracy: 57.5%
 
    Works when: the narrative structure causes frequent tone alternation.
    Fails when: a long non-story post also has many shifts, or a short story maintains 
    a consistent emotional tone.
    """

    scores = get_sentence_sentiments(text)
    if count_sentiment_shifts(scores) >= 5:
        return 'story'
    return 'non-story'

def sentence_count_decision(text: str) -> str:
    """
    Rule 3: Sentence count.
 
    Observation: Stories averaged 14.3 sentences vs 10.6 for non-stories
    on the development set. Stories need space to develop a plot.
    Rule: if sentence count >= 12, predict 'story'.
    Dev accuracy: 62.1%
 
    Works when: the text is long because it tells a narrative.
    Fails when: a long informational post exceeds the threshold, 
    or a short but clearly story-structured post falls below it.
    """

    scores = get_sentence_sentiments(text)
    if len(scores) >= 12:
        return 'story'
    return 'non-story'

#Combined decision (voting)

def determine_pragmatic_story(text: str) -> str:
    """
    Combine all three pragmatic rules using majority voting.
 
    Two or more votes for 'story' → predict 'story', otherwise 'non-story'.
    Dev accuracy: 60.1%
 
    The voting method reduces the impact of any single weak rule and
    provides more balanced performance across both classes.
    """

    votes = 0
    if emotional_range_decision(text) == 'story':
        votes += 1
    if sentiment_shift_decision(text) == 'story':
        votes += 1
    if sentence_count_decision(text) == 'story':
        votes +1 
    return 'story' if votes >= 2 else 'non-story'


#Main: calculating and pattern reporting

DEV_CSV = "dev.csv"

def main():
    """
    Evaluates all three pragmatic rules individually and as a combined voter on the data. 
    Prints observed averages and accuracy per rule.
    """

    total = 0
    stories = 0

    range_correct = 0 
    shifts_correct = 0 
    sents_correct = 0 
    combined_correct = 0

    story_ranges, nonstory_ranges = [], []
    story_shifts, nonstory_shifts = [], []
    story_sents, nonstory_sents = [], []

    df = pd.read_csv(DEV_CSV)
    for _, row in df.itterrows():
        text = row['content']
        label = row['label']

        sent_scores = get_sentence_sentiments(text)
        er = emotional_range(sent_scores)
        sh = count_sentiment_shifts(sent_scores)
        ns = len(sent_scores)

        if emotional_range_decision(text) == label:
            range_correct += 1
        if sentiment_shift_decision(text) == label:
            shifts_correct += 1
        if sentence_count_decision(text) == label:
            sents_correct += 1
        if determine_pragmatic_story(text) == label:
            combined_correct += 1

        if label == 'story': 
            stories += 1
            story_ranges.append(er)
            story_shifts.append(sh)
            story_sents.append(ns)
        else:
            nonstory_ranges.append(er)
            nonstory_shifts.append(sh)
            nonstory_sents.append(ns)

        total += 1

    non_stories = total - stories

    print(f"percentage emotional range correct: {round(range_correct / total * 100, 1)}")
    print(f"average emotional range for story: {sum(story_ranges) / stories:.2f}")
    print(f"average emotional range for non-story: {sum(nonstory_ranges) / non_stories:.2f}")
 
    print(f"percentage sentiment shifts correct: {round(shifts_correct / total * 100, 1)}")
    print(f"average sentiment shifts for story: {sum(story_shifts) / stories:.2f}")
    print(f"average sentiment shifts for non-story: {sum(nonstory_shifts) / non_stories:.2f}")
 
    print(f"percentage sentence count correct: {round(sents_correct / total * 100, 1)}")
    print(f"average sentence count for story: {sum(story_sents) / stories:.2f}")
    print(f"average sentence count for non-story: {sum(nonstory_sents) / non_stories:.2f}")
 
    print(f"percentage all pragmatic rules correct: {round(combined_correct / total * 100, 1)}")
 
 
if __name__ == "__main__":
    main()