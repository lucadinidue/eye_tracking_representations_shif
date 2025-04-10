import sys
import os
sys.path.append(os.path.abspath('.'))

from utils.dataset_utils import create_senteces_from_data, scale_datasets, tokenize_and_align_labels
from utils.custom_data_collator import DataCollatorForMultiTaskTokenClassification
from transformers import TrainingArguments, AutoTokenizer, Trainer, AutoConfig
from utils.custom_modeling_bert import BertForMultitaskTokenClassification
import pandas as pd
import evaluate
import argparse

TASKS = ['firstfix_dur','dur','firstrun_nfix','nfix','firstrun_dur']

def preprocess_dataset(dataset_path, model_name):
    df = pd.read_csv(dataset_path, index_col=0)
    dataset = create_senteces_from_data(df, TASKS)

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    tokenized_dataset = dataset.map(
                    tokenize_and_align_labels(tokenizer,[f'label_{task}' for task in TASKS]),
                    batched=True,
                    remove_columns=['text'],
                    desc="Running tokenizer on dataset",
                    )
    
    data_collator = DataCollatorForMultiTaskTokenClassification(tokenizer)
    return tokenized_dataset, data_collator

mae = evaluate.load('mae')
spearmanr = evaluate.load("spearmanr")
def compute_metrics(eval_pred):
    res = dict()
    for task_idx, task in enumerate(TASKS):
        labels = eval_pred.label_ids[task_idx].flatten()
        predictions = eval_pred.predictions[task].squeeze().flatten()
        
        not_masked_labels = labels != -100
        labels = labels[not_masked_labels]
        predictions = predictions[not_masked_labels]
        res[task] = {
            'mae': mae.compute(predictions=predictions, references=labels)['mae'],
            'spearmanr': spearmanr.compute(predictions=predictions, references=labels)['spearmanr']
        }
    return res

def train_model(args,train_dataset, data_collator):
    config = AutoConfig.from_pretrained(args.model_name)
    config.update({'tasks': TASKS, 'keys_to_ignore_at_inference':['mse_loss', 'labels', 'tasks_loss']})
    model = BertForMultitaskTokenClassification.from_pretrained(args.model_name, config=config)

    training_args = TrainingArguments(
            output_dir=args.output_directory, 
            eval_strategy='epoch',
            logging_strategy='epoch',
            label_names=[f'label_{task}' for task in TASKS],
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.training_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=0.05,
            save_strategy = 'no'
            )
    
    # We evaluate on training data, the dataset is too small to extract a test set
    # moreover we are not actually interested on model's performance
    train_dataset, test_dataset = scale_datasets(train_dataset, train_dataset)

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    trainer.train()
    trainer.evaluate()
    trainer.save_model()
    trainer.save_state()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', dest='model_name', type=str, default='FacebookAI/xlm-roberta-base')
    parser.add_argument('-o', '--output_directory', dest='output_directory', type=str)
    parser.add_argument('-u', '--user_id', type=int)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=5e-05)
    parser.add_argument('-e', '--epochs', dest='training_epochs', type=int, default=200)
    parser.add_argument('-d', '--weight_decay', dest='weight_decay', type=float, default=0.01)
    args = parser.parse_args()

    dataset_path = f'data/meco/meco_users/it_{args.user_id}.csv'

    user_dataset, data_collator = preprocess_dataset(dataset_path, args.model_name)
    train_model(args, user_dataset, data_collator)


if __name__ == '__main__':
    main()