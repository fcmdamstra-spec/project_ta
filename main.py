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

# The high-level labels: three combined predictions plus the three analysis labels.
OVERALL_COLUMNS = [
    'combined_weighted', 'combined_unweighted', 'combined_modules',
    'syntax', 'semantics', 'pragmatic',
]

# The 10 individual sub-rules. The combined label is a weighted vote over these.
SUBRULE_COLUMNS = [
    'syntax_pron', 'syntax_propn', 'syntax_dep',
    'coref', 'ner', 'wordnet_noun', 'ambiguous',
    'emotional_range', 'sentiment_shift', 'sentence_count',
]

# Weight (1-5) each rule gets when voting. Higher = more trusted.
# Starting values reflect each rule's standalone accuracy; tune as needed.
RULE_WEIGHTS = {
    'syntax_pron': 3,
    'syntax_propn': 1,
    'syntax_dep': 3,
    'coref': 5,
    'ner': 2,
    'wordnet_noun': 3,
    'ambiguous': 5,
    'emotional_range': 1,
    'sentiment_shift': 2,
    'sentence_count': 4,
}

# A text is a 'story' when at least half of the total weight votes 'story'.
STORY_WEIGHT_THRESHOLD = sum(RULE_WEIGHTS.values()) / 2


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

        # Combined label: weighted vote over all 10 sub-rules (not the 3 analysis labels).
        subrules = [
            syn['syntax_pron'], syn['syntax_propn'], syn['syntax_dep'],
            sem['coref'], sem['ner'], sem['wordnet_noun'], sem['ambiguous'],
            pra['emotional_range'], pra['sentiment_shift'], pra['sentence_count'],
        ]
        # 1: weighted vote over the 10 sub-rules (a tie counts as story).
        story_weight = sum(
            RULE_WEIGHTS[name]
            for name, vote in zip(SUBRULE_COLUMNS, subrules)
            if vote == 'story'
        )
        combined_weighted = 'story' if story_weight >= STORY_WEIGHT_THRESHOLD else 'non-story'

        # 2: unweighted vote over the 10 sub-rules (5+ of 10 -> story).
        combined_unweighted = 'story' if subrules.count('story') >= 5 else 'non-story'

        # 3: unweighted vote over the 3 module labels (2+ of 3 -> story).
        module_labels = [syn['syntax'], sem['semantics'], pra['pragmatic']]
        combined_modules = 'story' if module_labels.count('story') >= 2 else 'non-story'

        rows.append({
            'text': text,
            'true_label': label,
            'combined_weighted': combined_weighted,
            'combined_unweighted': combined_unweighted,
            'combined_modules': combined_modules,
            'syntax': syn['syntax'],
            'semantics': sem['semantics'],
            'pragmatic': pra['pragmatic'],
            'syntax_pron': syn['syntax_pron'],
            'syntax_propn': syn['syntax_propn'],
            'syntax_dep': syn['syntax_dep'],
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