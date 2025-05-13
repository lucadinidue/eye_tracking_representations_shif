import pandas as pd
import argparse
import os

EXCLUDED_ENGLISH_SENTENCES = ['2.0_7.0', '3.0_10.0'] # excluded since the users that read most of the sentences did not read these sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', type=str, choices=['en', 'it'])
    args = parser.parse_args()

    meco_users_dir = f'data/meco/{args.language}'
    
    user_read_sentences = dict()
    all_sentences_ids = set()

    for file_name in os.listdir(meco_users_dir):
        user_id = int(file_name.split('_')[1][:-len('.csv')])
        file_path = os.path.join(meco_users_dir, file_name)
        user_df = pd.read_csv(file_path, index_col=0)
        grouped_user_df = user_df.groupby(['trialid', 'sentnum']) # each item is a sentence

        read_sentences = []
        for (trial_id, sentnum), _ in grouped_user_df:
            all_sentences_ids.add(f'{trial_id}_{sentnum}')
            read_sentences.append(f'{trial_id}_{sentnum}')
        
        user_read_sentences[user_id] = read_sentences

    if args.language == 'en':
       filtered_sentences = [sentence_id for sentence_id in all_sentences_ids if sentence_id not in EXCLUDED_ENGLISH_SENTENCES]
       filtered_users = []
       for user_id, read_sentences in user_read_sentences.items():
           missing = False
           for sentence_id in filtered_sentences:
               if sentence_id not in read_sentences:
                   missing = True
           if not missing:
               filtered_users.append(user_id)
    else:
        num_sentences = len(list(all_sentences_ids))
        filtered_users = [user_id for user_id, read_sentences in user_read_sentences.items() if len(read_sentences) == num_sentences]
        
    print('Users who read all sentences:')
    print(sorted(filtered_users))
       
if __name__ == '__main__':
    main()