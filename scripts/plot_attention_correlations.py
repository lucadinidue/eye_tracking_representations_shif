from utils.dataset_utils import create_senteces_from_data
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import json
import os

sns.set_style('darkgrid')

USERS = [26, 3, 11, 13, 17, 21, 26, 36, 37, 48]

def load_json(src_path):
    with open(src_path, 'r') as src_file:
        loaded_dict = json.load(src_file)
    return loaded_dict

def load_eye_tracking_data(src_path, col):
    data = pd.read_csv(src_path, index_col=0)
    gaze_dataset = create_senteces_from_data(data, [col], keep_id=True)
    return gaze_dataset

def compute_layer_correlation(eye_tracking_data, layer_attentions, eye_tracking_feature):
    human_attentions = []
    model_attentions = []
    for sentence in eye_tracking_data:
        human_attentions += sentence[f'label_{eye_tracking_feature}']
        model_attentions += layer_attentions[sentence['id']]
    corr = spearmanr(human_attentions, model_attentions)
    if corr.pvalue < 0.05:
        return max(corr.statistic, -corr.statistic)
    else:
        return None

def compute_layers_correlation(correlations_dict, attention_dir, eye_tracking_data, user_id, eye_tracking_feature, model_name):
    for layer in range(12):
        layer_path = os.path.join(attention_dir, f'{layer}.json')
        layer_dict = load_json(layer_path)
        corr = compute_layer_correlation(eye_tracking_data, layer_dict, eye_tracking_feature)

        correlations_dict['user'].append(user_id)
        correlations_dict['layer'].append(layer)
        correlations_dict['score'].append(corr)
        correlations_dict['model'].append(model_name)


def compute_correlations_df(attention_dir, baseline_attention_dir, eye_tracking_dir, eye_tracking_feature):
    correlations_dict = {'user': [], 'layer':[], 'score': [], 'model':[]}

    for user_id in USERS:
        user_attention_dir = os.path.join(attention_dir, f'user_{user_id}')
        user_baseline_attention_dir = os.path.join(baseline_attention_dir, f'user_{user_id}')
        user_eye_tracking_path = os.path.join(eye_tracking_dir, f'it_{user_id}.csv')

        eye_tracking_data = load_eye_tracking_data(user_eye_tracking_path, eye_tracking_feature)
        
        compute_layers_correlation(correlations_dict, user_attention_dir, eye_tracking_data, user_id, eye_tracking_feature, 'finetuned')
        compute_layers_correlation(correlations_dict, user_baseline_attention_dir, eye_tracking_data, user_id, eye_tracking_feature, 'baseline')

    return pd.DataFrame.from_dict(correlations_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eye_tracking_feature', type=str, default='dur')
    args = parser.parse_args()

    attention_dir = f'data/attentions/finetuned'
    baseline_attention_dir = f'data/attentions/baseline'
    eye_tracking_dir = 'data/meco/meco_users'
    output_dir = 'data/results/attention_correlation'

    correlations_df = compute_correlations_df(attention_dir, baseline_attention_dir, eye_tracking_dir, args.eye_tracking_feature)
    sns.lineplot(data=correlations_df, x='layer', y='score', hue='model', palette='tab10', marker='o')
    plt.savefig(os.path.join(output_dir, 'avg_correlations.png'), bbox_inches='tight') 
    plt.cla()

    for user_id in sorted(correlations_df['user'].unique()):
        user_df = correlations_df[correlations_df['user'] == user_id]    
        sns.lineplot(data=user_df, x='layer', y='score', hue='model', palette='tab10', marker='o').set_title(f'User {user_id}')
        plt.savefig(os.path.join(output_dir, f'{user_id}.png'), bbox_inches='tight') 
        plt.cla()


if __name__ == '__main__':
    main()