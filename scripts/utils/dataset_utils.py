from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
import numpy as np
import json

def create_senteces_from_data(data: pd.DataFrame, tasks: list, keep_id:bool=False) -> Dataset:
    dropping_cols = set(data.columns).difference(set(tasks))
    
    # sort by trialid, sentnum and ianum, to avoid splitted sentences
    data = data.sort_values(by=["trialid", "sentnum", "ianum"])

    # create sentence_id
    data["sentence_id"] = data["trialid"].astype(int).astype(str) + '_' +data["sentnum"].astype(int).astype(str)
    
    dropping_cols.add("sentence_id")
    
    word_func = lambda s: [str(w) for w in s["ia"].values.tolist()]

    features_func = lambda s: [np.array(s.drop(columns=dropping_cols).iloc[i])
                            for i in range(len(s))]

    grouped_data = data.groupby("sentence_id")

    sentences_ids = list(grouped_data.groups.keys())
    sentences = grouped_data.apply(word_func).tolist()
    targets = grouped_data.apply(features_func).tolist()

    data_list = []
    
    for s_id, s, t in zip(sentences_ids, sentences, targets):
        if keep_id:
            data_list.append({
            **{"id": s_id},
            **{"text": s,},
            **{"label_"+str(l) : np.array(t)[:, i] for i, l in enumerate(tasks)}
        })
        else:
            data_list.append({
                **{"text": s,},
                **{"label_"+str(l) : np.array(t)[:, i] for i, l in enumerate(tasks)}
            })

    return Dataset.from_list(data_list)


def create_and_fit_sclers(train_dataset:Dataset) -> dict[MinMaxScaler]:
    # create and fit the scalers
    features = [col_name for col_name in train_dataset.column_names if col_name.startswith('label_')]

    scalers = {}
    for feat in features:
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaler.fit(np.array([word_feat for sentence in train_dataset[feat] for word_feat in sentence]).reshape(-1, 1))
        scalers[feat] = scaler

    return scalers

def scale_train_dataset(train_data: Dataset) -> {Dataset, Dataset}:
    features_scalers = create_and_fit_sclers(train_data)

    def minmaxscaling_function(row):
        for feature, scaler in features_scalers.items():
            row[feature] = scaler.transform(np.array(row[feature]).reshape(-1, 1)).squeeze().tolist()
        return row

    train_data = train_data.map(minmaxscaling_function)

    return train_data


def tokenize_and_align_labels(tokenizer:AutoTokenizer, features:list):
    def _tokenize_and_align_labels(dataset, label_all_tokens=False):
        tokenized_inputs = tokenizer(dataset['text'], max_length=256, padding=True, truncation=True, is_split_into_words=True)
        labels = dict()
        for feature_name in features:
            labels[feature_name] = list()
            for i, label in enumerate(dataset[feature_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        if label_all_tokens:
                            label_ids.append(label[word_idx])
                        else:
                            label_ids.append(-100)

                    previous_word_idx = word_idx

                labels[feature_name].append(label_ids)
            tokenized_inputs[feature_name] = labels[feature_name]
        return tokenized_inputs

    return _tokenize_and_align_labels

def align_to_original_words(model_tokens: list, original_tokens: list, subword_prefix: str,
                            lowercase: bool = False) -> list:
    if lowercase:
        original_tokens = [tok.lower() for tok in original_tokens]
    model_tokens = model_tokens[1: -1]  # Remove <s> and </s>
    aligned_model_tokens = []
    alignment_ids = []
    alignment_id = -1
    orig_idx = 0
    for token in model_tokens:
        alignment_id += 1

        if token.startswith(subword_prefix):  # Remove the sub-word prefix
            token = token[len(subword_prefix):]
        if len(aligned_model_tokens) == 0:  # First token (serve?)
            aligned_model_tokens.append(token)
        elif aligned_model_tokens[-1] + token in original_tokens[orig_idx]:  # We are in the second (third, fourth, ...) sub-token
            aligned_model_tokens[-1] += token  # so we merge the token with its preceding(s)
            alignment_id -= 1
        elif aligned_model_tokens[-1] +' '+token in original_tokens[orig_idx]: # Sometimes there is a space in the original tokens
            aligned_model_tokens[-1] += ' '+token 
            alignment_id -= 1
        else:
            aligned_model_tokens.append(token)
        if aligned_model_tokens[-1] == original_tokens[orig_idx]:  # A token was equal to an entire original word or a set of
            orig_idx += 1  # sub-tokens was merged and matched an original word
        alignment_ids.append(alignment_id)

    if aligned_model_tokens != original_tokens:
        raise Exception(
            f'Failed to align tokens.\nOriginal tokens: {original_tokens}\nObtained alignment: {aligned_model_tokens}')
    return alignment_ids


def create_subwords_alignment(dataset: Dataset, tokenizer: AutoTokenizer, subword_prefix: str,
                              lowercase: bool = False) -> dict:
    sentence_alignment_dict = dict()

    for sent_id, sentence in zip(dataset['id'], dataset['text']):
        for tok_id, tok in enumerate(sentence):
            if tok == '–':
                sentence[tok_id] = '-'
            if 'ö' in tok:
                sentence[tok_id] = tok.replace('ö', '')
            if 'ô' in tok:
                sentence[tok_id] = tok.replace('ô', '')
            if 'û' in tok:
                sentence[tok_id] = tok.replace('û', '-')
        if lowercase:
            sentence = [word.lower() for word in sentence]
        tokenized_sentence = tokenizer(sentence, is_split_into_words=True, return_tensors='pt')
        input_ids = tokenized_sentence['input_ids'].tolist()[0]  # 0 because the batch_size is 1
        model_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        alignment_ids = align_to_original_words(model_tokens, sentence, subword_prefix, lowercase)
        sentence_alignment_dict[sent_id] = {'model_input': tokenized_sentence, 'alignment_ids': alignment_ids}
    return sentence_alignment_dict


def save_dictionary(dictionary, out_path):
    with open(out_path, 'w') as out_file:
        out_file.write(json.dumps(dictionary))