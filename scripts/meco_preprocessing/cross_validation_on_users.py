import sys
import os
sys.path.append(os.path.abspath('.'))

from utils.dataset_utils import create_senteces_from_data, scale_datasets, tokenize_and_align_labels
from utils.custom_modeling_roberta import RobertaForMultiTaskTokenClassification
from utils.custom_data_collator import DataCollatorForMultiTaskTokenClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, AutoConfig
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import evaluate
import argparse

TASKS = ['firstfix_dur','dur','firstrun_nfix','nfix','firstrun_dur']

def k_fold_split(dataset, k=5, seed=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    dataset_indices = np.arange(len(dataset))
    
    for train_indices, test_indices in kf.split(dataset_indices):
        train_split = dataset.select(train_indices)
        test_split = dataset.select(test_indices)
        yield train_split, test_split

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

def get_results(trainer):
    # last_epoch = trainer.args.num_train_epochs
    results_dict = {'feature':[], 'epoch':[], 'score':[]}
    log_history = trainer.state.log_history
    for el in log_history:
        if 'eval_loss' in el:
            epoch = el['epoch']
            for feature, metrics in el.items():
                if type(metrics) == dict:
                    results_dict['epoch'].append(epoch)
                    results_dict['feature'].append(feature[len('eval_'):])
                    results_dict['score'].append(metrics['spearmanr'])
    return results_dict

def evaluate_on_dataset(args, tokenized_dataset, data_collator, k=5):
    config = AutoConfig.from_pretrained(args.model_name)
    config.update({'tasks': TASKS, 'keys_to_ignore_at_inference':['mse_loss', 'labels', 'tasks_loss']})
    model = RobertaForMultiTaskTokenClassification.from_pretrained(args.model_name, config=config)

    training_args = TrainingArguments(
            output_dir='prova', 
            eval_strategy='epoch',
            logging_strategy='epoch',
            label_names=[f'label_{task}' for task in TASKS],
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.training_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            save_strategy = 'no',
            warmup_ratio=0.05
            )
    
    results = dict()
    for fold_idx, (train_dataset, test_dataset) in enumerate(k_fold_split(tokenized_dataset, k)):
        train_dataset, test_dataset = scale_datasets(train_dataset, test_dataset)

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
        results[fold_idx+1] = get_results(trainer)
    return results
    

def unroll_results_dict(user_id, user_results):
    columns = ['user_id', 'fold', 'epoch', 'feature', 'score']
    unrolled_dict = {col: [] for col in columns}

    for fold_idx, fold_dict in user_results.items():
        unrolled_dict['fold'] += [fold_idx] * len(fold_dict['feature'])
        for k, v in fold_dict.items():
            unrolled_dict[k] += v
    unrolled_dict['user_id'] = [user_id] * len(unrolled_dict['fold'])
    return unrolled_dict
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', dest='model_name', default='FacebookAI/xlm-roberta-base', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=5e-05)
    parser.add_argument('-e', '--epochs', dest='training_epochs', type=int, default=50)
    parser.add_argument('-d', '--weight_decay', dest='weight_decay', type=float, default=0.01)
    parser.add_argument('-k', '--k_fold', type=int, default=5)
    args = parser.parse_args()

    datasets_dir = 'data/meco/meco_users/'
    out_path = f'data/meco/cv_results_e{args.training_epochs}_lr{args.learning_rate}.csv'
    first_write = True if not os.path.exists(out_path) else False
    
    for user_file_name in sorted(os.listdir(datasets_dir)):        
        user_id = user_file_name.split('_')[1].split('.')[0]
        print(f'Processing user {user_id}')
        user_file_path = os.path.join(datasets_dir, user_file_name)
        user_dataset, data_collator = preprocess_dataset(user_file_path, args.model_name)
        user_results = evaluate_on_dataset(args, user_dataset, data_collator, args.k_fold)
        user_unrolled_results = unroll_results_dict(user_id, user_results)
        user_df = pd.DataFrame.from_dict(user_unrolled_results)
        user_df.to_csv(out_path, mode='a', header=first_write, index=False)
        first_write = False

if __name__ == '__main__':
    main()
