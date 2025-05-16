from collections import defaultdict
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm
import seaborn as sns
from utils import *
import pandas as pd
import numpy as np
import argparse
import json
import ast
import csv
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
    args = parser.parse_args()

    if args.language == 'it':
        ud_path = 'data/ud_treebank/it_isdt-ud-train_sentences.csv'
    else:
        ud_path = 'data/ud_treebank/en_ewt-ud-train_sentences.csv'
    baseline_attention_dir = f'data/attentions/{args.language}/ud/baseline'

    output_dir = f'data/results/{args.language}/attention_shift/{args.feature}'
    if args.pos_filter:
        output_dir = output_dir+f'_{args.pos_filter}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    user_ids = USER_IDS[args.language]
    all_attention_shifts_df, feature_occurences = load_attention_shift_df(user_ids, ud_path, baseline_attention_dir, args.language, output_dir, args.normalize, args.feature, args.pos_filter, args.max_plot_value, plot_user=True)

    output_path = os.path.join(output_dir, f'feature_distribution.png')
    plot_feature_occurences(feature_occurences, args.feature, output_path)

    output_path = os.path.join(output_dir, f'user_avg.png')
    all_attention_shifts_df = pd.concat(all_attention_shifts_df, ignore_index=True)
    plot_attention_shift(all_attention_shifts_df, args.max_plot_value, args.feature, output_path)
    

if __name__ == '__main__':
    main()