import pandas as pd
import argparse


class Token():

    def __init__(self, word):
        self.word = word
        self.length = len(word)
        self.index = None
        self.pos = None
        self.root_dist = None

    def set_features_from_line(self, splitted_line):
        self.pos = splitted_line[3]
        self.root_dist = int(splitted_line[6])


class Sentence():

    def __init__(self, sentence_id):
        self.sentence_id = sentence_id
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)
    
    def set_token_ids(self):
        for token_idx, token in enumerate(self.tokens):
            token.index = token_idx + 1

def load_sentences(src_path):
    sentences = []
    sentence = None
    lines_to_skip = 0       # per saltare righe dopo clitici o preposizioni articolate
    take_features = False   # le features delle parole splittate si prendono dalla prima sotto-parola
    
    for line in open(src_path):
        if line.startswith('# sent_id ='):
            if sentence:
                sentence.set_token_ids()
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
                token = Token(word=word)    # inizializzo il token solo con la forma
                sentence.add_token(token)
            else:
                if lines_to_skip == 0:              # parola normale
                    token = Token(word=word)
                    token.set_features_from_line(splitted_line)
                    sentence.add_token(token)
                if take_features:                   # sotto-parole
                    sentence.tokens[-1].set_features_from_line(splitted_line)   # prendo le features dalla prima sottoparola
                    take_features = False

                lines_to_skip = max(0, lines_to_skip-1)
    sentence.set_token_ids()    # rimane fuori l'ultima frase
    sentences.append(sentence)
    return sentences


def create_sentences_df(sentences, min_length):
    sentences_dict = {'sentence_id':[] , 'tokens':[], 'index':[], 'length':[], 'pos':[], 'root_dist':[]}

    for sentence in sentences:
        if sentence.sentence_id in ['tut-3572', 'tut-3587', '2_Europarl-194', '2_Europarl-200', '2_Europarl-247', '2_Europarl-265', 
                                    '2_Europarl-267']: # lead to error due to special characters    
            continue
        if len(sentence.tokens) >= min_length:
            sentences_dict['sentence_id'].append(sentence.sentence_id)
            sentences_dict['tokens'].append([token.word for token in sentence.tokens])
            sentences_dict['index'].append([token.index for token in sentence.tokens])
            sentences_dict['length'].append([token.length for token in sentence.tokens])
            sentences_dict['pos'].append([token.pos for token in sentence.tokens])
            sentences_dict['root_dist'].append([token.root_dist for token in sentence.tokens])

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