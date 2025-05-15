import os
import sys
sys.path.append(os.path.abspath(".")) 

from utils.dataset_utils import create_subwords_alignment, save_dictionary, get_subword_prefix
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils.attention_utils import get_attention_weights
from datasets import Dataset
import argparse
import torch
import csv
import ast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ud_sentences(src_path):
    records = []

    with open(src_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sentence = {
                'id': row['sentence_id'],
                'text': ast.literal_eval(row['tokens'])
            }
            records.append(sentence)

    dataset = Dataset.from_list(records)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', type=str, choices=['it', 'en'])
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-a', '--aggregation_method', type=str, choices=['cls', 'avg'], default='avg')
    args = parser.parse_args()

    if args.language == 'it':
        ud_path = 'data/ud_treebank/it_isdt-ud-train_sentences.csv'
    else:
        ud_path = 'data/ud_treebank/en_ewt-ud-train_sentences.csv'
    dataset = load_ud_sentences(ud_path)

    model_type = 'FacebookAI/xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_type, add_prefix_space=True)
    subword_prefix  = get_subword_prefix(model_type)

    sentence_alignment_dict = create_subwords_alignment(dataset, tokenizer, subword_prefix)

    model = AutoModelForMaskedLM.from_pretrained(args.model_path, output_attentions=True).to(device)
    attention_weights = get_attention_weights(model, sentence_alignment_dict, args.aggregation_method)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for layer in range(12):
        output_path = os.path.join(args.output_dir, f'{layer}.json')
        save_dictionary(attention_weights[layer], output_path)


if __name__ == '__main__':
    main()