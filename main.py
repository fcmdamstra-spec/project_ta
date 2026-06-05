'''
This file runs the pipeline of the different linguistics fields used to determine wheter something is a story or not
'''

import pandas as pd
import spacy
from semantics import semantics_breakdown
from pragmatics import pragmatics_breakdown
from syntaxis import syntax_breakdown

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('coreferee')

DEV_CSV = "dev.csv"
LOG_CSV = "analysis_log.csv"

# The 4 high-level labels: the final prediction plus the three analysis labels.
OVERALL_COLUMNS = ['combined', 'syntax', 'semantics', 'pragmatic']

# The 10 individual sub-rules. The combined label is a majority vote over these.
SUBRULE_COLUMNS = [
    'syntax_pron', 'syntax_noun_chunk', 'syntax_propn',
    'coref', 'ner', 'wordnet_noun', 'ambiguous',
    'emotional_range', 'sentiment_shift', 'sentence_count',
]


def main():

    '''
    This function combines all other python files, using a voting system to determine whether text is a story or not.

    For every text it records the true label, the combined prediction, each analysis label
    and each individual sub-rule decision. All rows are written to analysis_log.csv so the
    behaviour of every rule (and how they combine) can be inspected. It also prints the
    accuracy of every decision column to the console.
    '''

    rows = []

    df = pd.read_csv(DEV_CSV)
    for _, row in df.iterrows():
        text  = row['content']
        label = row['label']
        doc = nlp(text)

        syn = syntax_breakdown(doc)
        sem = semantics_breakdown(doc)
        pra = pragmatics_breakdown(text)

        # Combined label: majority vote over all 10 sub-rules (not the 3 analysis labels).
        subrules = [
            syn['syntax_pron'], syn['syntax_noun_chunk'], syn['syntax_propn'],
            sem['coref'], sem['ner'], sem['wordnet_noun'], sem['ambiguous'],
            pra['emotional_range'], pra['sentiment_shift'], pra['sentence_count'],
        ]
        story_votes = subrules.count('story')
        # 5 or more of the 10 rules say 'story' -> story (a 5-5 tie counts as story)
        combined = 'story' if story_votes >= 5 else 'non-story'

        rows.append({
            'text': text,
            'true_label': label,
            'combined': combined,
            'syntax': syn['syntax'],
            'semantics': sem['semantics'],
            'pragmatic': pra['pragmatic'],
            'syntax_pron': syn['syntax_pron'],
            'syntax_noun_chunk': syn['syntax_noun_chunk'],
            'syntax_propn': syn['syntax_propn'],
            'coref': sem['coref'],
            'ner': sem['ner'],
            'wordnet_noun': sem['wordnet_noun'],
            'ambiguous': sem['ambiguous'],
            'emotional_range': pra['emotional_range'],
            'sentiment_shift': pra['sentiment_shift'],
            'sentence_count': pra['sentence_count'],
        })

    log = pd.DataFrame(rows)
    log.to_csv(LOG_CSV, index=False)
    print(f'wrote {len(log)} rows to {LOG_CSV}\n')

    # Accuracy of the high-level labels against the true label.
    print('--- Overall ---')
    for col in OVERALL_COLUMNS:
        accuracy = (log[col] == log['true_label']).mean()
        print(f'{col + ":":<20}{round(accuracy * 100, 1)}%')

    # Accuracy of each individual sub-rule against the true label.
    print('\n--- Sub-rules ---')
    for col in SUBRULE_COLUMNS:
        accuracy = (log[col] == log['true_label']).mean()
        print(f'{col + ":":<20}{round(accuracy * 100, 1)}%')


if __name__ == "__main__":
    main()
