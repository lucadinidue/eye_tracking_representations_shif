from __future__ import annotations
from typing import List
import pandas as pd
import argparse

######## DA FINIRE!!!


class Token():

    def __init__(self, index:int, word:str):
        
        # Token info
        self.token_idx = index
        self.word = word
        self.head_idx = None
       
        # Token features
        self.length = len(word)
        self.pos = None
        self.head_dist = None
        self.relation_type = None
        self.ariety = 0


    def set_features_from_line(self, splitted_line: List[str]):
        self.pos = splitted_line[3]
        self.head_idx = int(splitted_line[6])
        self.head_dist = self.head_idx - self.token_idx if self.head_idx != 0 else 0
        self.relation_type = splitted_line[7]

    def __str__(self):
        return f'{self.token_idx}\t{self.word}\t{self.head_dist}'



class Sentence():

    def __init__(self, sentence_id:str):
        self.sentence_id = sentence_id
        self.tokens = dict[Token]()

    def add_token(self, token:Token):
        self.tokens[token.token_idx] = token    

    def set_tokens_sentence_level_features(self):
        for token in self.tokens.values():
            if token.head_idx != 0 and token.pos != 'PUNCT':
                try:
                    self.tokens[token.head_idx].ariety += 1
                except:
                    self.tokens[token.head_idx-1].ariety += 1 # Possibile errore di annotazione?
    

def load_sentences(src_path):
    sentences = []
    sentence = None
    lines_to_skip = 0       # per saltare righe dopo clitici o preposizioni articolate
    take_features = False   # le features delle parole splittate si prendono dalla prima sotto-parola
    
    for line in open(src_path):
        if line.startswith('# sent_id ='):
            if sentence:
                sentence.set_tokens_sentence_level_features()
                sentences.append(sentence)
            sent_id = line[len('# sent_id = '):].strip()
            sentence = Sentence(sent_id)

        if line[0].isdigit():
            splitted_line = line.strip().split('\t')
            word = splitted_line[1]

            if '.' in splitted_line[0]: # caso speciale ?
                pass
            elif '-' in splitted_line[0]: # abbiamo trovato una parola splittata
                word_ids = splitted_line[0].split('-')
                lines_to_skip = int(word_ids[1]) - int(word_ids[0]) + 1 
                take_features = True 
                token = Token(index=int(word_ids[0]),word=word)    # inizializzo il token solo con la forma
                sentence.add_token(token)
            else:
                if lines_to_skip == 0:              # parola normale
                    token = Token(index=int(splitted_line[0]), word=word)
                    token.set_features_from_line(splitted_line)
                    sentence.add_token(token)
                if take_features:                   # sotto-parole
                    sentence.tokens[list(sentence.tokens.keys())[-1]].set_features_from_line(splitted_line)   # prendo le features dalla prima sottoparola
                    take_features = False

                lines_to_skip = max(0, lines_to_skip-1)
    sentence.set_tokens_sentence_level_features()    # rimane fuori l'ultima frase
    sentences.append(sentence)
    return sentences


def create_sentences_df(sentences, min_length):
    sentences_dict = {'sentence_id':[] , 'tokens':[], 'index':[], 'length':[], 'pos':[], 'head_dist':[], 'relation_type':[], 'ariety': []}

    for sentence in sentences:
        if sentence.sentence_id in ['tut-3572', 'tut-3587', '2_Europarl-194', '2_Europarl-200', '2_Europarl-247', '2_Europarl-265', 
                                    '2_Europarl-267']: # lead to error due to special characters    
            continue
        if len(sentence.tokens) >= min_length:
            sentence_tokens = list(sentence.tokens.values())
            sentences_dict['sentence_id'].append(sentence.sentence_id)
            sentences_dict['tokens'].append([token.word for token in sentence_tokens])
            sentences_dict['index'].append([token.token_idx for token in sentence_tokens])
            sentences_dict['length'].append([token.length for token in sentence_tokens])
            sentences_dict['pos'].append([token.pos for token in sentence_tokens])
            sentences_dict['head_dist'].append([token.head_dist for token in sentence_tokens])
            sentences_dict['relation_type'].append([token.relation_type for token in sentence_tokens])
            sentences_dict['ariety'].append([token.ariety for token in sentence_tokens])

    return pd.DataFrame.from_dict(sentences_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--min_length', type=int, default=6)
    args =  parser.parse_args()


    src_path = 'data/ud_treebank/it_isdt-ud-train.conllu'
    out_path = 'data/ud_treebank/it_isdt-ud-train_sentences.csv'

    sentences = load_sentences(src_path)
    sentences_df = create_sentences_df(sentences, min_length=args.min_length)
    sentences_df.to_csv(out_path)


if __name__ == '__main__':
    main()