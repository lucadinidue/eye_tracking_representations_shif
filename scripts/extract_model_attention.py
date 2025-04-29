import os
import sys
sys.path.append(os.path.abspath(".")) 

import torch

from utils.dataset_utils import create_subwords_alignment, save_dictionary, create_senteces_from_data, get_subword_prefix
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils.attention_utils import get_attention_weights
from tqdm import tqdm
import pandas as pd
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
USERS = [3, 11, 13, 17, 21, 26, 36, 37, 48]
    

def extract_attention_from_user(dataset_path, model_path, aggregation_method, output_dir):
    df = pd.read_csv(dataset_path, index_col=0)
    dataset = create_senteces_from_data(df, [], keep_id=True)

    model = AutoModelForMaskedLM.from_pretrained(model_path, output_attentions=True).to(device)

    model_type = 'FacebookAI/xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_type, add_prefix_space=True)
    subword_prefix = get_subword_prefix(model_type)

    sentence_alignment_dict = create_subwords_alignment(dataset, tokenizer, subword_prefix)
    attention_weights = get_attention_weights(model, sentence_alignment_dict, aggregation_method)

    for layer in range(12):
        output_path = os.path.join(output_dir, f'{layer}.json')
        save_dictionary(attention_weights[layer], output_path)

def extract_baseline_attention(aggregation_method):
    model_path = 'FacebookAI/xlm-roberta-base'

    for user_id in USERS:
        dataset_path = f'data/meco/meco_users/it_{user_id}.csv'
        output_dir = f'data/attentions/baseline/user_{user_id}'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        extract_attention_from_user(dataset_path, model_path, aggregation_method, output_dir)


def extract_model_attention(user_id, training_config, aggregation_method):
    dataset_path = f'data/meco/meco_users/it_{user_id}.csv'
    model_path = f'models/{training_config}/user_{user_id}'
    output_dir = f'data/attentions/{training_config}/user_{user_id}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extract_attention_from_user(dataset_path, model_path, aggregation_method, output_dir)    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user_id', type=str)
    parser.add_argument('-c', '--training_config', type=str)
    parser.add_argument('-b', '--baseline', action='store_true')
    parser.add_argument('-a', '--aggregation_method', type=str, choices=['avg', 'cls'], default='cls')
    args = parser.parse_args()

    if args.baseline:
        extract_baseline_attention(args.aggregation_method)
    else:
        extract_model_attention(args.user_id, args.training_config, args.aggregation_method)


if __name__ == '__main__':
    main()