"""
Part 3 - Pragmatics Analysis
Created by Frederique, s5042968

Classifies Reddit posts as 'story' or 'non-story' using three pragmatic features
based on sentence-level sentiment analysis (VADER via asent):

1. Emotional range = difference between the most positive and negative sentence
2. Sentiment shift = number of times tone changes across sentences
3. Sentence count = stories tend to be longer and use more sentences

Each feature produces one rule. The three rules vote: 2 or more votes -> predicted story

Observation from dev data: 
    Stories averaged an emotional range of 1.14 vs 1.04 for non-stories
    Stories averaged 5.60 sentiment shifts vs 4.46 for non-stories
    Stories averaged 14.26 sentences vs 10.62 for non-stories
"""

import spacy
import pandas as pd
import asent

nlp_sentiment = spacy.blank('en')
nlp_sentiment.add_pipe('sentencizer')
nlp_sentiment.add_pipe('asent_en_v1')


# Extraction helpers

def get_sentence_sentiments(text: str) -> list[float]:
    """
    Return a list of compound sentiment scores, one per sentence.
    The compound score ranges from -1 to +1. A score > 0.05 is considered positive, 
    <-0.05 negative and in between neutral. 
    = standard VADER thresholds. 
    """
    doc = nlp_sentiment(text)
    return [sent._.polarity.compound for sent in doc.sents]


def emotional_range(sent_scores: list[float]) -> float:
    """
    The difference between the highest and lowest sentence-level compound scores.
    """
    if len(sent_scores) < 2:
        return 0.0
    return max(sent_scores) - min(sent_scores)


def count_sentiment_shifts(sent_scores: list[float]) -> int:
    """
    Count how many times the sentiment direction changes between consecutive sentences.
    A shift is counted when a sentence's polarity category (positive / non-positive)
    differs from the previous sentence.
    """
    if len(sent_scores) < 2:
        return 0
    shifts = 0
    for i in range(1, len(sent_scores)):
        if (sent_scores[i] > 0.05) != (sent_scores[i - 1] > 0.05):
            shifts += 1
    return shifts


# Individual decision rules

def emotional_range_decision(text: str) -> str:
    scores = get_sentence_sentiments(text)
    if emotional_range(scores) >= 0.9:
        return 'story'
    return 'non-story'


def sentiment_shift_decision(text: str) -> str:
    scores = get_sentence_sentiments(text)
    if count_sentiment_shifts(scores) >= 5:
        return 'story'
    return 'non-story'


def sentence_count_decision(text: str) -> str:
    scores = get_sentence_sentiments(text)
    if len(scores) >= 12:
        return 'story'
    return 'non-story'


# Combined decision (voting)

def pragmatics_breakdown(text: str) -> dict:
    """
    Returns each pragmatic sub-rule decision plus the combined pragmatic label.
    The 3 rules vote; 2 or more 'story' votes makes the combined label 'story'.
    Single source of truth used by both determine_pragmatic_story and the logging
    in main.py.
    """
    decisions = {
        'emotional_range': emotional_range_decision(text),
        'sentiment_shift': sentiment_shift_decision(text),
        'sentence_count': sentence_count_decision(text),
    }
    votes = list(decisions.values()).count('story')
    decisions['pragmatic'] = 'story' if votes >= 2 else 'non-story'
    return decisions


def determine_pragmatic_story(text: str) -> str:
    """
    Combine all three pragmatic rules using majority voting.
    Two or more votes for 'story' -> predict 'story', otherwise 'non-story'.
    """
    return pragmatics_breakdown(text)['pragmatic']


# Main: calibration and pattern reporting

DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"

def main():
    """
    Evaluates all three pragmatic rules individually and as a combined voter on the data.
    Prints observed averages and accuracy per rule, then writes pragmatic_patterns.txt.
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

    # collect false positives and negatives for each rule
    fp_range, fn_range = [], []
    fp_shifts, fn_shifts = [], []
    fp_sents, fn_sents = [], []

    choice = input("Run on (d)ev or (t)est dataset? [d/t]: ").strip().lower()
    csv_file = DEV_CSV if choice == 'd' else TEST_CSV
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        text = row['content']
        label = row['label']

        sent_scores = get_sentence_sentiments(text)
        er = emotional_range(sent_scores)
        sh = count_sentiment_shifts(sent_scores)
        ns = len(sent_scores)

        pred_range = emotional_range_decision(text)
        pred_shifts = sentiment_shift_decision(text)
        pred_sents = sentence_count_decision(text)

        if pred_range == label:
            range_correct += 1
        elif pred_range == 'story':
            fp_range.append(text)
        else:
            fn_range.append(text)

        if pred_shifts == label:
            shifts_correct += 1
        elif pred_shifts == 'story':
            fp_shifts.append(text)
        else:
            fn_shifts.append(text)

        if pred_sents == label:
            sents_correct += 1
        elif pred_sents == 'story':
            fp_sents.append(text)
        else:
            fn_sents.append(text)

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

    with open("pragmatic_patterns.txt", "w", encoding="utf-8") as f:
        f.write(f"""
Pragmatic patterns

1. Emotional range
Observation: stories averaged 1.14 vs 1.04 for non-stories.
Method:      asent (VADER) -> emotional_range = max sentence compound score - min sentence compound score
Rule:        emotional_range >= 0.9 -> predict 'story'
Accuracy:    {round(range_correct/total*100, 1)}%
Works when:  the text contains clearly positive and negative moments, e.g. a personal narrative with both conflict and resolution.
Fails when:  opinionated non-story posts swing between praise and criticism, producing a wide range without narrative structure.
             Example false positive: "{fp_range[0][:200].replace(chr(10), ' ')}..."
             Stories told in a single consistent tone fall below the threshold.
             Example false negative: "{fn_range[0][:200].replace(chr(10), ' ')}..."

2. Sentiment shifts
Observation: stories averaged 5.60 shifts vs 4.46 for non-stories.
Method:      asent (VADER) -> count of consecutive sentences changing between positive and non-positive
Rule:        sentiment_shifts >= 5 -> predict 'story'
Accuracy:    {round(shifts_correct/total*100, 1)}%
Works when:  the narrative alternates between positive and negative moments, e.g. a personal story with both hardship and relief.
Fails when:  long non-story posts move between subtopics with different tones, producing many shifts without narrative structure.
             Example false positive: "{fp_shifts[0][:200].replace(chr(10), ' ')}..."
             Short or tonally consistent stories do not accumulate enough shifts.
             Example false negative: "{fn_shifts[0][:200].replace(chr(10), ' ')}..."

3. Sentence count
Observation: stories averaged 14.26 sentences vs 10.62 for non-stories.
Method:      asent sentencizer -> number of sentences in the text
Rule:        sentence_count >= 12 -> predict 'story'
Accuracy:    {round(sents_correct/total*100, 1)}%
Works when:  the text is long because it describes a sequence of events.
Fails when:  long informational posts such as detailed arguments exceed 12 sentences without any narrative.
             Example false positive: "{fp_sents[0][:200].replace(chr(10), ' ')}..."
             Short but complete stories with fewer than 12 sentences are missed.
             Example false negative: "{fn_sents[0][:200].replace(chr(10), ' ')}..."

Combined (2+ of 3 rules vote 'story'): {round(combined_correct/total*100, 1)}%
""")
    print("wrote pragmatic_patterns.txt")


if __name__ == "__main__":
    main()