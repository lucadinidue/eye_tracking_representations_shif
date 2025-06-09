from utils import *
from scipy.stats import mannwhitneyu
import argparse

import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'axes.facecolor': '#f5f5f5',
    'figure.facecolor': '#e0e0e0',
    'axes.edgecolor': '#cccccc',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'font.size': 9,
    'legend.fontsize': 8
})

color_incr = '#1a9850'  # verde intenso
color_decr = '#d73027'  # rosso mattone
color_line = '#999999'  # grigio medio

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

def get_shift_significance(attention_distribution_baseline, attention_distribution_finetuned):
    significance_dict = {layer: dict() for layer in range(12)}
    for layer in attention_distribution_baseline.keys():
        for feat_value in sorted(attention_distribution_baseline[layer]):
            distribution_baseline = attention_distribution_baseline[layer][feat_value]
            distribution_finetuned = attention_distribution_finetuned[layer][feat_value]
            difference = np.mean(distribution_finetuned) - np.mean(distribution_baseline)
            direction = 'I' if difference > 0 else 'D'
            _, p_val = mannwhitneyu(distribution_baseline, distribution_finetuned)
            significance = 'S' if p_val < 0.01 else 'U'
            significance_dict[layer][feat_value] = {'s': significance, 'd': direction}
    return significance_dict

def get_attention_distributions(dataset, feature, baseline_attentions, finetuned_attentions):
    attention_distribution_baseline = {layer: defaultdict(list) for layer in baseline_attentions.keys()}
    attention_distribution_finetuned = {layer: defaultdict(list) for layer in baseline_attentions.keys()}
    for layer in baseline_attentions.keys():
        for el in dataset:
            sentence_id = el['id']
            feature_values = el[feature]
            baseline_sentence_attention = baseline_attentions[layer][sentence_id]
            finetuned_sentence_attention = finetuned_attentions[layer][sentence_id]
            for feat_value, b_att, f_att in zip(feature_values, baseline_sentence_attention, finetuned_sentence_attention):
                attention_distribution_baseline[layer][feat_value].append(b_att)
                attention_distribution_finetuned[layer][feat_value].append(f_att)
    return attention_distribution_baseline, attention_distribution_finetuned

def update_significance_dict(all_significance, user_significance):
    for layer in user_significance.keys():
        for feat_value in user_significance[layer].keys():
            if feat_value not in all_significance[layer]:
                all_significance[layer][feat_value] = {'I': 0, 'D':0}
            if user_significance[layer][feat_value]['s'] == 'S':
                direction = user_significance[layer][feat_value]['d']
                all_significance[layer][feat_value][direction] += 1


def normalize_significance_dict(all_significance, num_users):
    for layer in all_significance.keys():
        for feat_value in all_significance[layer].keys():
            all_significance[layer][feat_value]['I'] /= num_users
            all_significance[layer][feat_value]['D'] /= num_users


def filter_features(all_significance, feature):
    if feature == 'pos':
        all_features = list(set(list(all_significance[0].keys())) - set(['INTJ', 'X', 'INTJ', 'PART']))
    else:
        all_features = [feat_value for feat_value in all_significance[0].keys() if feat_value <= 15]
        all_features = [feat_value for feat_value in all_features if feat_value > -15]
    return sorted(all_features)

def plot_significances(all_significance, language, feature):
    all_features = filter_features(all_significance, feature)

    fig, axes = plt.subplots(nrows=len(all_features), figsize=(8, 12), sharex=True)

    for idx, feat_value in enumerate(all_features):
        x, y_decr, y_incr = [], [], []
        for layer in range(12):
            x.append(layer+1)
            y_decr.append(all_significance[layer][feat_value]['D'])
            y_incr.append(all_significance[layer][feat_value]['I'])

        axes[idx].plot(x, y_decr, label='Decreased', color=color_decr, linewidth=1.5)
        axes[idx].plot(x, y_incr, label='Increased', color=color_incr, linewidth=1.5)
        
        axes[idx].hlines([0.5], xmin=1, xmax=12, color=color_line, linestyle='--', linewidth=0.8)
        
        axes[idx].set_xlim((1,12))
        axes[idx].set_ylim((-0.03, 1.03))
        axes[idx].set_ylabel(f'{feat_value}', rotation=0, labelpad=20, ha='right', fontsize=9)
        axes[idx].set_yticks([])
        axes[idx].grid(True, axis='y', linestyle=':', linewidth=0.5)

        if idx != len(all_features) - 1:
            axes[idx].tick_params(axis='x', colors='#f5f5f5')

    axes[-1].set_xticks(range(1, 13))
    axes[-1].set_xlabel('Layer')

    # Legenda
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.93, 0.01), frameon=False)

    plt.subplots_adjust(hspace=0.15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.suptitle(f'Significance of attention shift for {LABEL_MAP[feature]}', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(f'data/results/{language}/significance', f'{feature}.png'))
    plt.show()


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

    dataset = load_ud_dataset(ud_path)
    baseline_attentions = {layer: load_json(os.path.join(baseline_attention_dir, f'{layer}.json')) for layer in range(12)}

    all_significance = {layer: dict() for layer in range(12)}
    for user_id in user_ids:
        finetuned_attention_dir = f'data/attentions/{args.language}/ud/finetuned/user_{user_id}'
        finetuned_attentions = {layer: load_json(os.path.join(finetuned_attention_dir, f'{layer}.json')) for layer in range(12)}
        attention_distribution_baseline, attention_distribution_finetuned = get_attention_distributions(dataset, args.feature, baseline_attentions, finetuned_attentions)
        user_significance = get_shift_significance(attention_distribution_baseline, attention_distribution_finetuned)
        update_significance_dict(all_significance, user_significance)
    
    normalize_significance_dict(all_significance, len(user_ids))
    plot_significances(all_significance, args.language, args.feature)

if __name__ == '__main__':
    main()