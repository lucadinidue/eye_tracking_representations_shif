import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import *
import pandas as pd
import argparse
import os

sns.set_style('darkgrid')

USER_IDS = {
    'it':[3, 11, 13, 17, 21, 26, 36, 37, 48],
    'en': [3, 6, 72, 74, 76, 78, 79, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 97, 98, 99, 101, 102]
} 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', type=str, choices=['it', 'en'])
    parser.add_argument('-f', '--feature', type=str)
    parser.add_argument('-m', '--max_plot_value', type=int)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-p', '--pos_filter', type=str)
    parser.add_argument('-r', '--relation_filter', type=str)

    args = parser.parse_args()
    
    baseline_attention_dir = f'../data/attentions/{args.language}/ud/baseline'

    if args.language == 'it':
        ud_path = 'data/ud_treebank/it_isdt-ud-train_sentences.csv'
    else:
        ud_path = 'data/ud_treebank/en_ewt-ud-train_sentences.csv'
    baseline_attention_dir = f'data/attentions/{args.language}/ud/baseline'

    output_dir = f'data/results/{args.language}/attention_shift/{args.feature}'
    if args.pos_filter:
        output_dir = output_dir+f'_{args.pos_filter}'
    if args.relation_filter:
        output_dir = output_dir+f'_{args.relation_filter}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    user_ids = USER_IDS[args.language]
    all_attention_shifts_df, _ = load_attention_shift_df(user_ids, ud_path, baseline_attention_dir, args.language, output_dir, args.normalize, args.feature, args.pos_filter, args.relaion_filter, args.max_plot_value, plot_user=False)

    difference_df = all_attention_shifts_df[all_attention_shifts_df['model'] == 'difference']

    output_path = os.path.join(output_dir, 'layers_average.png')
    plt.figure(figsize=(20, 10))
    p = sns.barplot(data=difference_df, x='feature_value', y='attention')
    p.set_title(f'Attention shift for {args.feature} averaged across layers')
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.clf()

    output_path = os.path.join(output_dir, 'differences_across_layers.png')
    plt.figure(figsize=(20, 10))
    p = sns.lineplot(data=difference_df, x='layer', y='attention', hue='feature_value')
    p.set_title(f'Attention shift for {args.feature}')
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.clf()

    # Absolute values of difference
    output_path = os.path.join(output_dir, 'differences_across_layers_abs.png')
    difference_df['attention'] = difference_df['attention'].abs()
    sns.lineplot(data=difference_df, x='layer', y='attention', hue='feature_value')
    p.set_title(f'Attention shift for {args.feature}')
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

if __name__ == '__main__': 
    main()