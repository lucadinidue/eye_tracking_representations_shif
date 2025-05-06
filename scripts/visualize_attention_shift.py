import matplotlib.pyplot as plt
from datasets import Dataset
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import json
import ast
import csv
import os

sns.set_style('darkgrid')


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


def add_layer_attention_to_dataset(dataset, baseline_attention_dir, finetuned_attention_dir, layer, normalize_attention):
    baseline_layer_path = os.path.join(baseline_attention_dir, f'{layer}.json')
    finetuned_layer_path = os.path.join(finetuned_attention_dir, f'{layer}.json')
    baseline_attention = load_json(baseline_layer_path)
    finetuned_attention = load_json(finetuned_layer_path)

    def add_attention_columns(example):
        sent_id = example['id']
        if normalize_attention:
            example['baseline_attention'] = normalize_list(baseline_attention[sent_id])
            example['finetuned_attention'] = normalize_list(finetuned_attention[sent_id])
        else:
            example['baseline_attention'] = baseline_attention[sent_id]
            example['finetuned_attention'] = finetuned_attention[sent_id]
        return example

    dataset = dataset.map(add_attention_columns)
    return dataset


def sum_attention_across_dataset(dataset, feature):
    attention_counter_baseline = {}
    attention_counter_finetuned = {}
    feature_occurrences = {}

    for el in dataset:
        assert len(el[feature]) == len(el['baseline_attention']) == len(el['finetuned_attention'])
        for feature_value, attention_bl, attention_ft in zip(el[feature], el['baseline_attention'], el['finetuned_attention']):
            if feature_value not in attention_counter_baseline:
                attention_counter_baseline[feature_value] = 0
                attention_counter_finetuned[feature_value] = 0
                feature_occurrences[feature_value] = 0
            attention_counter_baseline[feature_value] += attention_bl
            attention_counter_finetuned[feature_value] += attention_ft
            feature_occurrences[feature_value] += 1
    for feature_value, num_occurrences in feature_occurrences.items():
        attention_counter_baseline[feature_value] /= num_occurrences
        attention_counter_finetuned[feature_value] /= num_occurrences
    return attention_counter_baseline, attention_counter_finetuned


def compute_attention_shift(attention_counter_baseline, attention_counter_finetuned, add_difference=True):
    attention_shift = {'feature_value': [], 'attention': [], 'model':[]}

    for feat_value in attention_counter_baseline.keys():
        attention_shift['feature_value'].append(feat_value)
        attention_shift['attention'].append(attention_counter_baseline[feat_value])
        attention_shift['model'].append('baseline')
    
        attention_shift['feature_value'].append(feat_value)
        attention_shift['attention'].append(attention_counter_finetuned[feat_value])
        attention_shift['model'].append('finetuned')

        if add_difference:
            attention_shift['feature_value'].append(feat_value)
            attention_shift['attention'].append(attention_counter_finetuned[feat_value] - attention_counter_baseline[feat_value])
            attention_shift['model'].append('difference')

    attention_shift = pd.DataFrame.from_dict(attention_shift)
    return attention_shift

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user_id', type=int)
    parser.add_argument('-f', '--feature', type=str)
    parser.add_argument('-m', '--max_plot_value', type=int)
    parser.add_argument('-n', '--normalize', action='store_true')
    args = parser.parse_args()

    ud_path = 'data/ud_treebank/it_isdt-ud-train_sentences.csv'
    baseline_attention_dir = 'data/attentions/ud/baseline'
    finetuned_attention_dir = f'data/attentions/ud/finetuned/user_{args.user_id}'

    output_dir = f'data/results/attention_shift/{args.feature}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'user_{args.user_id}.png')

    dataset = load_ud_dataset(ud_path)

    fig, axes = plt.subplots(12, 1, figsize=(10, 50))

    for layer in range(12):
        dataset = add_layer_attention_to_dataset(dataset, baseline_attention_dir, finetuned_attention_dir, layer, args.normalize)
        attention_counter_baseline, attention_counter_finetuned = sum_attention_across_dataset(dataset, args.feature) 
        attention_shift = compute_attention_shift(attention_counter_baseline, attention_counter_finetuned, add_difference=True)

        if args.max_plot_value:
            attention_shift = attention_shift[attention_shift['feature_value'] <= args.max_plot_value]

        sns.barplot(
            data=attention_shift,
            x='feature_value',
            y='attention',
            hue='model',
            palette='muted',
            ax=axes[layer]
        )
        
        axes[layer].set_title(f'Feature = {args.feature}, Layer = {layer+1}')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


if __name__ == '__main__':
    main()