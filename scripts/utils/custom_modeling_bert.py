# Extended from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
"""PyTorch BERT model."""

from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.utils import logging

from transformers import BertPreTrainedModel, BertModel

logger = logging.get_logger(__name__)



import torch
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import logging, ModelOutput



def mask_loss(b_output, b_target, target_pad):
    """
    Masks the pad tokens of by setting the corresponding output and target tokens equal.
    """
    active_outputs = b_output.view(-1)
    active_targets = b_target.view(-1)
    active_mask = active_targets == target_pad

    active_outputs = torch.where(active_mask, active_targets, active_outputs)

    return active_outputs, active_targets


@dataclass
class MultiTaskTokenClassifierOutput(ModelOutput):
    """
    Class for outputs of multitask token classification models.

    Args:
        loss (float):
            average MSE loss.
        tasks_loss (dict) :
            MSE loss for each task.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the selfattention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    tasks_loss: Optional[dict] = None
    logits: Tuple[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    labels: Optional[dict] = None



class BertForMultitaskTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # classifiers
        self.tasks = config.tasks
        self.num_tasks = len(self.tasks)
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(config.hidden_size, 1) for
            task in self.tasks
        })

        # Initialize weights and apply final processing
        self.post_init()

    

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **labels_list # contains eye-tracking labels
    ) -> Union[Tuple[torch.Tensor], MultiTaskTokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        labels = dict()
        for key, value in labels_list.items():
            if key.startswith('label_'):
                labels[key[len('label_'):]] = value


        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        
        loss = 0
        tasks_loss = {}
        logits = {}

        for task in self.tasks:
            task_logits = self.classifiers[task](sequence_output)
            logits[task] = task_logits

            if labels[task] is not None:
                task_labels = labels[task].to(task_logits.device)
                output_, target_ = mask_loss(task_logits, task_labels, -100)

                # MSE Loss
                loss_fct = MSELoss()
                task_loss = loss_fct(output_, target_)
                tasks_loss[task] = task_loss
                loss += task_loss

        loss /= self.num_tasks

        return MultiTaskTokenClassifierOutput(
            loss=loss,
            tasks_loss=tasks_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            labels=labels
        )
    