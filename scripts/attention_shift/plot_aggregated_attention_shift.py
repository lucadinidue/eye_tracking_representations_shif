import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
import argparse


USER_IDS = {
    'it':[3, 11, 13, 17, 21, 26, 36, 37, 48],
    'en': [3, 6, 72, 74, 76, 78, 79, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 97, 98, 99, 101, 102]
} 

LABEL_MAP = {
    'index': 'Word poisition in sentence',
    'head_dist': 'Distance from syntactic head',
    'pos': 'Part of Speech',
    'length': 'Word length'
}

def filter_df_by_plot_values(difference_df, feature):
    if feature == 'index':
        filtered_df = difference_df[difference_df['feature_value'] <= 25]
    elif feature in ['head_dist', 'length']:
        filtered_df = difference_df[difference_df['feature_value'] <= 15]
        filtered_df = filtered_df[filtered_df['feature_value'] >= -15]
    elif feature == 'pos':
        filtered_df = difference_df[~difference_df['feature_value'].isin(['INTJ', 'X', 'INTJ', 'PART'])]
    return filtered_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', type=str, choices=['it', 'en'])
    parser.add_argument('-f', '--feature', type=str)
    parser.add_argument('-m', '--max_plot_value', type=int)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-p', '--pos_filter', nargs='+', default=[])
    parser.add_argument('-r', '--relation_filter', type=str)
    args = parser.parse_args()

    baseline_attention_dir = f'data/attentions/{args.language}/ud/baseline'
    if args.language == 'it':
        ud_path = 'data/ud_treebank/it_isdt-ud-train_sentences.csv'
    else:
        ud_path = 'data/ud_treebank/en_ewt-ud-train_sentences.csv'
    user_ids = USER_IDS[args.language]

    output_path = f'data/results/{args.language}/attention_shift/heatmaps/{args.feature}.png'

    all_attention_shifts_df, _ = load_attention_shift_df(user_ids, ud_path, baseline_attention_dir, args.language, None, True, args.feature, args.pos_filter, args.relation_filter, None, plot_user=False)
    difference_df = all_attention_shifts_df[all_attention_shifts_df['model'] == 'difference']
    difference_df['layer'] = difference_df['layer']+1

    filtered_df = filter_df_by_plot_values(difference_df, args.feature)

    pivot_df = filtered_df.pivot_table(index='feature_value', columns='layer', values='attention')
    pivot_df['AVG'] = pivot_df.mean(axis=1)

    plt.figure(figsize=(10, 8))
    p = sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap='coolwarm', center=0.00, cbar=False)
    p.axvline([12] ,color='white',linestyle='-')
    p.set_yticklabels(p.get_yticklabels(), rotation=0);
    p.set(ylabel=LABEL_MAP[args.feature], xlabel='Layer');
    language = 'Italian' if args.language == 'it' else 'English'
    p.set_title(language)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()