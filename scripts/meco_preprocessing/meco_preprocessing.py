import pandas as pd
import argparse
import os

COLS_TO_KEEP = ['uniform_id', 'trialid', 'sentnum', 'ianum', 'ia', 'firstfix.dur', 'firstrun.dur', 'dur', 'firstrun.nfix', 'nfix']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, choices=['en', 'it'])
    args = parser.parse_args()
                                     
    dataset_path = 'data/meco/joint_data_trimmed.csv'
    output_dir = f'data/meco/{args.language}'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    df = pd.read_csv(dataset_path, index_col=0)
    df = df[df['lang'] == args.language]     #  take only italian/english subset of the dataset
    df.loc[:, 'ia'] = df['ia'].fillna('[UNK]')      # 'ia' is the read word, if is NaN put the unknown token of the model tokenizer
    df = df.fillna(0.0)             # the NaN values correspond to skipped words

    user_dfs = df.groupby('uniform_id')
    for user_id, user_df in user_dfs:
        user_df = user_df[COLS_TO_KEEP]
        user_df.rename(columns={k: "_".join(k.split(".")) for k in COLS_TO_KEEP}, inplace=True)
        output_path = os.path.join(output_dir, f'{user_id}.csv')
        user_df.to_csv(output_path)

if __name__ == '__main__':
    main()