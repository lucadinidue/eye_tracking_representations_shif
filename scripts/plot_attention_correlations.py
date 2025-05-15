from utils.dataset_utils import create_senteces_from_data
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import json
import os

sns.set_style('darkgrid')

USERS = {
    'it':[3, 11, 13, 17, 21, 26, 36, 37, 48],
    'en': [3, 6, 72, 74, 76, 78, 79, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 97, 98, 99, 101, 102]
} 
def load_json(src_path):
    with open(src_path, 'r') as src_file:
        loaded_dict = json.load(src_file)
    return loaded_dict

def load_eye_tracking_data(src_path, col):
    data = pd.read_csv(src_path, index_col=0)
    gaze_dataset = create_senteces_from_data(data, [col], keep_id=True)
    return gaze_dataset

def compute_layer_correlation(eye_tracking_data, layer_attentions, eye_tracking_feature, allow_negative_scores):
    human_attentions = []
    model_attentions = []
    for sentence in eye_tracking_data:
        human_attentions += sentence[f'label_{eye_tracking_feature}']
        model_attentions += layer_attentions[sentence['id']]
    corr = spearmanr(human_attentions, model_attentions)
    if corr.pvalue < 0.05:
        if allow_negative_scores:
            return corr.statistic
        return max(corr.statistic, -corr.statistic)
    else:
        return None

def compute_layers_correlation(correlations_dict, attention_dir, eye_tracking_data, user_id, eye_tracking_feature, model_name, allow_negative_scores):
    for layer in range(12):
        layer_path = os.path.join(attention_dir, f'{layer}.json')
        layer_dict = load_json(layer_path)
        corr = compute_layer_correlation(eye_tracking_data, layer_dict, eye_tracking_feature, allow_negative_scores)

        correlations_dict['user'].append(user_id)
        correlations_dict['layer'].append(layer+1)
        correlations_dict['score'].append(corr)
        correlations_dict['model'].append(model_name)


def compute_correlations_df(attention_dir, baseline_attention_dir, eye_tracking_dir, eye_tracking_feature, allow_negative_scores, language):
    correlations_dict = {'user': [], 'layer':[], 'score': [], 'model':[]}

    for user_id in USERS[language]:
        user_attention_dir = os.path.join(attention_dir, f'user_{user_id}')
        user_baseline_attention_dir = os.path.join(baseline_attention_dir, f'user_{user_id}')
        user_eye_tracking_path = os.path.join(eye_tracking_dir, f'{language}_{user_id}.csv')

        eye_tracking_data = load_eye_tracking_data(user_eye_tracking_path, eye_tracking_feature)
        
        compute_layers_correlation(correlations_dict, user_attention_dir, eye_tracking_data, user_id, eye_tracking_feature, 'finetuned', allow_negative_scores)
        compute_layers_correlation(correlations_dict, user_baseline_attention_dir, eye_tracking_data, user_id, eye_tracking_feature, 'baseline', allow_negative_scores)

    return pd.DataFrame.from_dict(correlations_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', type=str, choices=['en', 'it'])
    parser.add_argument('-e', '--eye_tracking_feature', type=str, default='dur')
    parser.add_argument('-n', '--allow_negative_scores', action='store_true')
    args = parser.parse_args()

    attention_dir = f'data/attentions/{args.language}/meco/finetuned'
    baseline_attention_dir = f'data/attentions/{args.language}/meco/baseline'
    eye_tracking_dir = f'data/meco/{args.language}'
    output_dir = f'data/results/{args.language}/attention_correlation'
    if args.allow_negative_scores:
        output_dir += '_negatives'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    correlations_df = compute_correlations_df(attention_dir, baseline_attention_dir, eye_tracking_dir, args.eye_tracking_feature, args.allow_negative_scores, args.language)
    sns.lineplot(data=correlations_df, x='layer', y='score', hue='model', palette='tab10', marker='o')
    plt.axhline(0, color='black', linewidth=1)
    plt.savefig(os.path.join(output_dir, 'avg_correlations.png'), bbox_inches='tight') 
    plt.cla()

    for user_id in sorted(correlations_df['user'].unique()):
        user_df = correlations_df[correlations_df['user'] == user_id]    
        sns.lineplot(data=user_df, x='layer', y='score', hue='model', palette='tab10', marker='o').set_title(f'User {user_id}')
        plt.axhline(0, color='black', linewidth=1)
        plt.savefig(os.path.join(output_dir, f'{user_id}.png'), bbox_inches='tight') 
        plt.cla()


if __name__ == '__main__':
    main()