from collections import defaultdict
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import json
import ast
import csv
import os

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


def sum_attention_across_dataset(dataset, feature, pos_filter, relation_filter):
    attention_counter_baseline = [defaultdict(float) for _ in range(12)]
    attention_counter_finetuned = [defaultdict(float) for _ in range(12)]
    feature_occurrences = defaultdict(int)

    for el in dataset:
        for i, feature_value in enumerate(el[feature]):
            if not pos_filter or el['pos'][i] in pos_filter:
                if not relation_filter or el['relation_type'][i] == relation_filter: 
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

    return attention_counter_baseline, attention_counter_finetuned, feature_occurrences



def compute_layer_attention_shift(attention_counter_baseline, attention_counter_finetuned, add_difference=True):
    layer_records = []

    for feat_value in attention_counter_baseline.keys():
        layer_records.append({'feature_value': feat_value, 'attention': attention_counter_baseline[feat_value], 'model': 'baseline'})

        layer_records.append({'feature_value': feat_value, 'attention': attention_counter_finetuned[feat_value], 'model': 'finetuned'})

        if add_difference:
            difference =  attention_counter_finetuned[feat_value] - attention_counter_baseline[feat_value]
            relative_difference = difference / attention_counter_baseline[feat_value]
            layer_records.append({'feature_value': feat_value, 
                            'attention': relative_difference, 
                            'model': 'difference'})

    layer_df = pd.DataFrame.from_records(layer_records)
    return layer_df



def compute_user_shift(dataset, baseline_attentions, finetuned_attentions, normalize, feature, pos_filter, relation_filter):

    dataset = add_attentions_to_dataset(dataset, baseline_attentions, finetuned_attentions, normalize)
    attention_counter_baseline, attention_counter_finetuned, feature_occurences = sum_attention_across_dataset(dataset, feature, pos_filter, relation_filter) 


    all_layers_df = []
    for layer in range(12):
        layer_baseline = attention_counter_baseline[layer]
        layer_fnetuned = attention_counter_finetuned[layer]
        layer_df = compute_layer_attention_shift(layer_baseline, layer_fnetuned, add_difference=True)
        layer_df['layer'] = layer
        all_layers_df.append(layer_df)

    return pd.concat(all_layers_df, ignore_index=True), feature_occurences


def plot_attention_shift(attention_shift, max_plot_value, feature_name, output_path):
    _, axes = plt.subplots(12, 1, figsize=(10, 50))

    if max_plot_value:
        attention_shift = attention_shift[attention_shift['feature_value'] <= max_plot_value]
        attention_shift = attention_shift[attention_shift['feature_value'] >= -max_plot_value]

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

def load_attention_shift_df(user_ids, ud_path, baseline_attention_dir, language, output_dir, normalize, feature, pos_filter, relation_filter, max_plot_value=None, plot_user=False):
    dataset = load_ud_dataset(ud_path)
    baseline_attentions = {layer: load_json(os.path.join(baseline_attention_dir, f'{layer}.json')) for layer in range(12)}

    all_attention_shifts = []
    with tqdm(total=len(user_ids)) as pbar:
        for user_id in user_ids:
            finetuned_attention_dir = f'data/attentions/{language}/ud/finetuned/user_{user_id}'
            output_path = os.path.join(output_dir, f'user_{user_id}.png')
            finetuned_attentions = {layer: load_json(os.path.join(finetuned_attention_dir, f'{layer}.json')) for layer in range(12)}
            attention_shift, feature_occurences = compute_user_shift(dataset, baseline_attentions, finetuned_attentions, normalize, feature, pos_filter, relation_filter)
            if plot_user:
                plot_attention_shift(attention_shift, max_plot_value, feature, output_path)
            attention_shift['user'] = user_id
            all_attention_shifts.append(attention_shift)
            pbar.update(1)
    all_attention_shifts_df = pd.concat(all_attention_shifts, ignore_index=True)
    return all_attention_shifts_df, feature_occurences

def plot_feature_occurences(feature_occurences, feature_name, output_path):
    x = list(feature_occurences.keys())
    y = list(feature_occurences.values())
    plt.bar(x, y)
    plt.xticks(rotation=90)
    plt.title(f'Values distribution for feature: {feature_name}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()