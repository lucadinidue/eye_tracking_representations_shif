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
    
USERS = {
    'it':[3, 11, 13, 17, 21, 26, 36, 37, 48],
    'en': [3, 6, 72, 74, 76, 78, 79, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 97, 98, 99, 101, 102]
} 
    

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

def extract_baseline_attention(aggregation_method, language):
    model_path = 'FacebookAI/xlm-roberta-base'

    for user_id in USERS[language]:
        dataset_path = f'data/meco/{language}/{language}_{user_id}.csv'
        output_dir = f'data/attentions/{language}/meco/baseline/user_{user_id}'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        extract_attention_from_user(dataset_path, model_path, aggregation_method, output_dir)


def extract_model_attention(user_id, aggregation_method, language):
    dataset_path = f'data/meco/{language}/{language}_{user_id}.csv'
    model_path = f'models/{language}/user_{user_id}'
    output_dir = f'data/attentions/{language}/meco/finetuned/user_{user_id}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extract_attention_from_user(dataset_path, model_path, aggregation_method, output_dir)    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', type=str, choices=['en', 'it'])
    parser.add_argument('-u', '--user_id', type=str)
    parser.add_argument('-c', '--training_config', type=str)
    parser.add_argument('-b', '--baseline', action='store_true')
    parser.add_argument('-a', '--aggregation_method', type=str, choices=['avg', 'cls'], default='cls')
    args = parser.parse_args()

    if args.baseline:
        extract_baseline_attention(args.aggregation_method, args.language)
    else:
        extract_model_attention(args.user_id, args.aggregation_method, args.language)


if __name__ == '__main__':
    main()