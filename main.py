'''
This file runs the pipeline of the different linguistics fields used to determine wheter something is a story or not
'''

import pandas as pd
import spacy
from semantics import determine_semantics_story

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('coreferee')

DEV_CSV = "dev.csv"


def main():

    '''
    This function combines all other python files, using a voting system to determine whether text is a story or not
    '''

    total = 0
    total_correct = 0
    combined_correct = 0

    syntax_correct = 0
    semantics_correct = 0
    pragmatic_correct = 0


    df = pd.read_csv(DEV_CSV)
    for _, row in df.iterrows():
        text  = row['content']
        label = row['label']
        doc = nlp(text)

        total += 1

        #syntax_story = determine_syntax_story(doc)

        semantics_story = determine_semantics_story(doc)

        #pragmatic_story = determine_pragmatic_story(doc)

        #if syntax_story == label:
        #    syntax_correct += 1
        #    total_correct += 1
        if semantics_story == label:
            semantics_correct += 1
            total_correct += 1
        #if pragmatic_story == label:
        #    pragmatic_correct += 1
        #    total_correct += 1
    print(f'percentage correct: {round(total_correct/total, 2)}')

if __name__ == "__main__":
    main()





