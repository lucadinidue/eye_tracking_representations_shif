from collections import defaultdict
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import json
import ast
import csv
import os

sns.set_style('darkgrid')

USER_IDS = [3, 11, 13, 17, 21, 26, 36, 37, 48]


def load_json(src_path):
    with open(src_path, 'r') as src_file:
        loaded_dict = json.load(src_file)
    return loaded_dict

def load_features_names(src_path):
    with open(src_path, 'r') as src_file:
        header = src_file.readline()
    features = header.strip().split(',')[3:]
    return features

def load_ud_dataset(src_path):
    features = load_features_names(src_path)
    records = []

    with open(src_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            
            sentence = {feat: ast.literal_eval(row[feat]) for feat in features}
            sentence['id'] =  row['sentence_id']
            sentence['text'] =  ast.literal_eval(row['tokens'])
            
            records.append(sentence)

    dataset = Dataset.from_list(records)
    return dataset


def normalize_list(l):
    a = np.array(l)
    return (a / a.sum()).tolist()
    # return ((l-np.min(l))/(np.max(l)-np.min(l))).tolist()


def add_attentions_to_dataset(dataset, baseline_attentions, finetuned_attentions, normalize_attention):
    def add_attention_columns(example):
        sent_id = example['id']
        for layer in range(12):
            baseline = baseline_attentions[layer][sent_id]
            finetuned = finetuned_attentions[layer][sent_id]
            assert len(baseline) == len(finetuned)
            if normalize_attention:
                baseline = normalize_list(baseline)
                finetuned = normalize_list(finetuned)
            example[f'baseline_attention_{layer}'] = baseline
            example[f'finetuned_attention_{layer}'] = finetuned
        return example

    dataset = dataset.map(add_attention_columns)
    return dataset


def sum_attention_across_dataset(dataset, feature, pos_filter):
    attention_counter_baseline = [defaultdict(float) for _ in range(12)]
    attention_counter_finetuned = [defaultdict(float) for _ in range(12)]
    feature_occurrences = defaultdict(int)

    for el in dataset:
        for i, feature_value in enumerate(el[feature]):
            if not pos_filter or el['pos'][i] == pos_filter:
                feature_occurrences[feature_value] += 1
                for layer in range(12):
                    attention_bl = el[f'baseline_attention_{layer}'][i]
                    attention_ft = el[f'finetuned_attention_{layer}'][i]
                    attention_counter_baseline[layer][feature_value] += attention_bl
                    attention_counter_finetuned[layer][feature_value] += attention_ft

    # Normalize over occurrences
    for layer in range(12):
        for feature_value in attention_counter_baseline[layer]:
            count = feature_occurrences[feature_value]
            attention_counter_baseline[layer][feature_value] /= count
            attention_counter_finetuned[layer][feature_value] /= count

    return attention_counter_baseline, attention_counter_finetuned



def compute_layer_attention_shift(attention_counter_baseline, attention_counter_finetuned, add_difference=True):
    layer_records = []

    for feat_value in attention_counter_baseline.keys():
        layer_records.append({'feature_value': feat_value, 'attention': attention_counter_baseline[feat_value], 'model': 'baseline'})

        layer_records.append({'feature_value': feat_value, 'attention': attention_counter_finetuned[feat_value], 'model': 'finetuned'})

        if add_difference:
            layer_records.append({'feature_value': feat_value, 
                            'attention': attention_counter_finetuned[feat_value] - attention_counter_baseline[feat_value], 
                            'model': 'difference'})

    layer_df = pd.DataFrame.from_records(layer_records)
    return layer_df


def compute_user_shift(dataset, baseline_attentions, finetuned_attentions, normalize, feature, pos_filter):

    dataset = add_attentions_to_dataset(dataset, baseline_attentions, finetuned_attentions, normalize)
    attention_counter_baseline, attention_counter_finetuned = sum_attention_across_dataset(dataset, feature, pos_filter) 

    all_layers_df = []
    for layer in range(12):
        layer_baseline = attention_counter_baseline[layer]
        layer_fnetuned = attention_counter_finetuned[layer]
        layer_df = compute_layer_attention_shift(layer_baseline, layer_fnetuned, add_difference=True)
        layer_df['layer'] = layer
        all_layers_df.append(layer_df)

    return pd.concat(all_layers_df, ignore_index=True)


def plot_attention_shift(attention_shift, max_plot_value, feature_name, output_path):
    _, axes = plt.subplots(12, 1, figsize=(10, 50))

    if max_plot_value:
        attention_shift = attention_shift[attention_shift['feature_value'] <= max_plot_value]

    for layer in list(attention_shift['layer'].unique()):
        layer_attention_shift = attention_shift[attention_shift['layer'] == layer]
        sns.barplot(
            data=layer_attention_shift,
            x='feature_value',
            y='attention',
            hue='model',
            palette='muted',
            ax=axes[layer]
        )
        
        axes[layer].set_title(f'Feature = {feature_name}, Layer = {layer+1}')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature', type=str)
    parser.add_argument('-m', '--max_plot_value', type=int)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-p', '--pos_filter', type=str)
    args = parser.parse_args()

    ud_path = 'data/ud_treebank/it_isdt-ud-train_sentences.csv'
    baseline_attention_dir = 'data/attentions/ud/baseline'

    output_dir = f'data/results/attention_shift/{args.feature}'
    if args.pos_filter:
        output_dir = output_dir+f'_{args.pos_filter}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = load_ud_dataset(ud_path)
    baseline_attentions = {layer: load_json(os.path.join(baseline_attention_dir, f'{layer}.json')) for layer in range(12)}

    all_attention_shifts = []
    with tqdm(total=len(USER_IDS)) as pbar:
        for user_id in USER_IDS:
            finetuned_attention_dir = f'data/attentions/ud/finetuned/user_{user_id}'
            output_path = os.path.join(output_dir, f'user_{user_id}.png')
            finetuned_attentions = {layer: load_json(os.path.join(finetuned_attention_dir, f'{layer}.json')) for layer in range(12)}
            attention_shift = compute_user_shift(dataset, baseline_attentions, finetuned_attentions, args.normalize, args.feature, args.pos_filter)
            plot_attention_shift(attention_shift, args.max_plot_value, args.feature, output_path)
            attention_shift['user'] = user_id
            all_attention_shifts.append(attention_shift)
            pbar.update(1)

    output_path = os.path.join(output_dir, f'user_avg.png')
    all_attention_shifts_df = pd.concat(all_attention_shifts, ignore_index=True)
    plot_attention_shift(all_attention_shifts_df, args.max_plot_value, args.feature, output_path)
    

if __name__ == '__main__':
    main()