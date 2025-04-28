import os
import sys
sys.path.append(os.path.abspath(".")) 

import torch

from utils.dataset_utils import create_subwords_alignment, save_dictionary, create_senteces_from_data
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import pandas as pd
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
USERS = [3, 11, 13, 17, 21, 26, 36, 37, 48]

def get_subword_prefix(tokenizer_name):
    if 'xlm-roberta' in tokenizer_name.lower():
        return '▁'
    if 'roberta' in tokenizer_name.lower():
        return'Ġ'
    else:
        raise Exception(f'Model {tokenizer_name} not supported yet.')
    
def aggregate_tokens_attention(sentence_attention:list, alignment_ids:list):
    sentence_attention = sentence_attention[1:-1] # remove <s> and </s>
    aggregated_weights = [[] for _ in range(len(set(alignment_ids)))]
    for al_id, att_weights in zip(alignment_ids, sentence_attention):
        aggregated_weights[al_id].append(att_weights)

    # si prende solo il primo token
    aggregated_weights = [el[0] for el in aggregated_weights]
    return aggregated_weights

def extract_sentence_attention(model: AutoModelForMaskedLM, tokenized_text:BatchEncoding, alignment_ids:list, sentence_aggregation_method:str) -> list:
    model_output = model(**tokenized_text)
    attention_matrices = model_output['attentions']
    layers_attentions = []
    for layer in range(len(attention_matrices)):
        layer_attention_matrix = attention_matrices[layer].detach().squeeze()
        avg_attention_matrix = torch.mean(layer_attention_matrix, dim=0) # media tra le teste di attenzione
        if sentence_aggregation_method == 'avg':
            sentence_attention = torch.mean(avg_attention_matrix, dim=0).tolist()
        elif sentence_aggregation_method == 'cls':
            sentence_attention = avg_attention_matrix[0].tolist()
        else:
            raise Exception(f'Method {sentence_aggregation_method} not implemented')
        sentence_attention = aggregate_tokens_attention(sentence_attention, alignment_ids)
        layers_attentions.append(sentence_attention)
    return layers_attentions


def get_attention_weights(model:AutoModelForMaskedLM, subwords_alignment_dict, sentence_aggregation_method):
    attention_weights = {layer:{} for layer in range(12)}
    for sent_id in tqdm(subwords_alignment_dict):
        tokenized_text = subwords_alignment_dict[sent_id]['model_input'].to(device)
        alignment_ids = subwords_alignment_dict[sent_id]['alignment_ids']
        layers_attentions = extract_sentence_attention(model, tokenized_text, alignment_ids, sentence_aggregation_method)
        for layer in range(len(layers_attentions)):
            attention_weights[layer][sent_id] = layers_attentions[layer]
    return attention_weights


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