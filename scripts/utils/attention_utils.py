from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoModelForMaskedLM
from tqdm import tqdm
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

