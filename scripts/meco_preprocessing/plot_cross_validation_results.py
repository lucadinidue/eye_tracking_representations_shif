import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file_name', type=str)
    args = parser.parse_args()

    input_path = os.path.join('data/cv_results/', args.input_file_name)
    output_path =input_path[:-len('csv')]+'png' 

    results_df = pd.read_csv(input_path)
    num_users = len(results_df['user_id'].unique())

    fig, axes = plt.subplots(nrows=num_users, figsize=(8, 5 * num_users), sharex=True)
    for ax, (user_id, user_df) in zip(axes,  results_df.groupby(['user_id'])):
        sns.lineplot(data=user_df, x='epoch', y='score', hue='feature', ax=ax)
        ax.set_title(f'User {user_id[0]}')

    plt.tight_layout()
    plt.savefig(output_path)

if __name__ == '__main__':
    main()