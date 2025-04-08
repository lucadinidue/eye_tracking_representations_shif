import pandas as pd
import os

def main():
    meco_users_dir = 'data/meco/meco_users'
    
    num_read_sentences = dict()
    all_sentences_ids = set()

    for file_name in os.listdir(meco_users_dir):
        user_id = int(file_name.split('_')[1][:-len('.csv')])
        file_path = os.path.join(meco_users_dir, file_name)
        user_df = pd.read_csv(file_path, index_col=0)
        grouped_user_df = user_df.groupby(['trialid', 'sentnum']) # each item is a sentence

        read_sentences = 0
        for (trial_id, sentnum), _ in grouped_user_df:
            all_sentences_ids.add(f'{trial_id}_{sentnum}')
            read_sentences += 1
        
        num_read_sentences[user_id] = read_sentences

    num_sentences = len(list(all_sentences_ids))
    filtered_users = [user_id for user_id, read_sentences in num_read_sentences.items() if read_sentences == num_sentences]
        
    print('Users who read all sentences:')
    print(sorted(filtered_users))
       
if __name__ == '__main__':
    main()