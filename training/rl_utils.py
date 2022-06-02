import torch
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def sequence_mask(lengths, max_len=None, dtype=None, device=None) :
    r"""Return a mask tensor representing the first N positions of each cell.
    If ``lengths`` has shape ``[d_1, d_2, ..., d_n]`` the resulting tensor
    ``mask`` has dtype ``dtype`` and shape ``[d_1, d_2, ..., d_n, maxlen]``,
    with
    ```
    mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
    ```
    Examples:
    ```python
    sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                 #  [True,  True,  True, False, False],
                                 #  [True,  True, False, False, False]]
    sequence_mask([[1, 3],[2,0]])  # [[[ True, False, False],
                                   #   [ True,  True,  True]],
                                   #  [[ True,  True, False],
                                   #   [False, False, False]]]
    ```
    Args:
        lengths: integer tensor or list of int, all its values <= max_len.
        max_len: scalar integer tensor, size of last dimension of returned
            tensor. Default is the maximum value in ``lengths``.
        dtype: the desired data type of returned tensor. Default: if None,
            returns :torch:`ByteTensor`.
        device: the desired device of returned tensor. Default: if None, uses
            the current device for the default tensor type.
    Returns:
        A mask tensor of shape :python:`lengths.shape + (max_len,)`, cast to
        specified dtype.
    Raises:
        ValueError: if ``max_len`` is not a scalar.
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device)
    elif device is None:
        device = lengths.device
    lengths: torch.LongTensor
    if max_len is None:
        max_len = torch.max(lengths).item()

    size = lengths.size()
    row_vector = torch.arange(max_len, device=device, dtype=lengths.dtype).view(
        *([1] * len(size)), -1).expand(*size, max_len)
    mask = (row_vector < lengths.unsqueeze(-1)).to(device=device)
    if dtype is not None:
        mask = mask.to(dtype=dtype)

    return mask


def masked_reverse_cumsum(X, lengths, dim):
    """
    Args:
        X (Tensor): [batch_size, max_tgt_len]
        lengths (Tensor): [batch_size]
        dim (int): -1
        gamma (float): the discount factor
    
    """
    masked_X = X * sequence_mask(lengths, max_len=X.shape[1])
    return (masked_X
            .flip(dims=[dim])
            .cumsum(dim=dim)
            .flip(dims=[dim]))


def discounted_future_sum(values, lengths, num_steps=None, gamma=1.0):
    """
    Args:
        values (Tensor): reward values with size [batch_size, max_tgt_len]
        lengths (Tensor): target sequence length with size [batch_size]
        num_steps (int): number of future steps to sum over.
        gamma (float): discount value.
    
    Return:
        output (Tensor): [batch_size, max_tgt_len]
    """
    assert values.dim() == 2
    
    batch_size, total_steps = values.shape
    values = values * sequence_mask(lengths, max_len=values.shape[1])

    num_steps = total_steps if num_steps is None else num_steps
    num_steps = min(num_steps, total_steps)
    
    padding = torch.zeros([batch_size, num_steps - 1]).to(values)
    padded_values = torch.cat([values, padding], 1)
    discount_filter = gamma ** torch.arange(num_steps).to(values).reshape(1, 1, -1)

    output = F.conv1d(padded_values.unsqueeze(-2), discount_filter).squeeze(1)
    return output


def polyak_update(model, tgt_model, target_lr):
    for param_, param in zip(tgt_model.parameters(), model.parameters()):
        param_.data.copy_((1 - target_lr) * param_ + target_lr * param)


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    reward_pad_value: float = 0.0
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        rewards = [feature["rewards"] for feature in features] if "rewards" in features[0].keys() else None
        if rewards is not None:
            max_reward_length = max(len(l) for l in rewards)
            if self.pad_to_multiple_of is not None:
                max_reward_length = (
                    (max_reward_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.reward_pad_value] * (max_reward_length - len(feature["rewards"]))
                if isinstance(feature["rewards"], list):
                    feature["rewards"] = (
                        feature["rewards"] + remainder if padding_side == "right" else remainder + feature["rewards"]
                    )
                elif padding_side == "right":
                    feature["rewards"] = np.concatenate([feature["rewards"], remainder])
                else:
                    feature["rewards"] = np.concatenate([remainder, feature["rewards"]])

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features