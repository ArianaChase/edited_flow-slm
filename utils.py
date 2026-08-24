import re
import os
import importlib.util
import torch
import lightning as pl
from pathlib import Path
import glob
import shutil
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
import json
from datasets import load_dataset, Audio
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from wordfreq import word_frequency
from tqdm import tqdm

def batch_pad_right(tensors: list, mode="constant", value=0):
    """
    COPY FROM SPEECHBRAIN
    Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Parameters
    ----------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.

    """

    if not len(tensors):
        raise IndexError("Tensors list must not be empty")

    if len(tensors) == 1:
        # if there is only one tensor in the batch we simply unsqueeze it.
        return tensors[0].unsqueeze(0), torch.tensor([1.0])

    if not (
        all(
            [tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))]
        )
    ):
        raise IndexError("All tensors must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the first dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(tensors[0].ndim):
        if dim != 0:
            if not all(
                [x.shape[dim] == tensors[0].shape[dim] for x in tensors[1:]]
            ):
                raise EnvironmentError(
                    "Tensors should have same dimensions except for the first one"
                )
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        # for each tensor we apply pad_right_to
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value
        )
        batched.append(padded)
        valid.append(valid_percent[0])

    batched = torch.stack(batched)

    return batched, torch.tensor(valid)


def pad_right_to(
    tensor: torch.Tensor, target_shape: (list, tuple), mode="constant", value=0,
):
    """
    COPY FROM SPEECHBRAIN
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Parameters
    ----------
    tensor : input torch tensor
        Input tensor whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode : str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value : float
        Pad value, please refer to torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == tensor.ndim
    pads = []  # this contains the abs length of the padding for each dimension.
    valid_vals = []  # this contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1

    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """
    COPY FROM SPEECHBRAIN
    Creates a binary mask for each sequence.

    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask

def select_latest_ckpt(ckpt_dir: str):
    ckpt_dir = Path(ckpt_dir)
    ckpts = sorted(glob.glob(str(ckpt_dir / "*.ckpt")), key=extract_number)
    hpc_ckpts = sorted(glob.glob(str(ckpt_dir / "hpc_ckpt_*.ckpt")), key=extract_number)
    # cleanup older hpc checkpoints (keep last)
    for p in hpc_ckpts[:-1]:
        try:
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)
        except FileNotFoundError:
            pass
    # choose most recent between last hpc and last ckpt
    candidate = None
    if hpc_ckpts and ckpts:
        try:
            if os.path.getmtime(hpc_ckpts[-1]) > os.path.getmtime(ckpts[-1]) and os.listdir(hpc_ckpts[-1]):
                candidate = "hpc"
            else:
                candidate = ckpts[-1]
        except OSError:
            candidate = ckpts[-1]
    elif hpc_ckpts:
        candidate = "hpc"
    elif ckpts:
        candidate = ckpts[-1]
    return candidate

class SaveAtSpecificStep(pl.Callback):
    def __init__(self, save_steps=100000, ckpt_dir=None):
        self.save_steps = save_steps
        self.ckpt_dir = ckpt_dir

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.save_steps == 0:
            print(f"Save checkpoint at step {trainer.global_step}")
            checkpoint_path = f"{self.ckpt_dir}/checkpoint_at_step_{trainer.global_step}.ckpt"
            trainer.save_checkpoint(checkpoint_path)

def is_overlapping(a_start, a_end, b_start, b_end):
    if (a_end >= b_start and a_start <= b_end):
        return True
    else:
        return False


def strip_stress(phone_label):
    if phone_label[-1].isdigit():
        return phone_label[:-1]
    else:
        return phone_label

def extract_timestamps(args, output):
    error_log = []
    losses = []

    for utterance_info in output:
        REDUCTION_FACTOR = 1
        FRAMERATE = 1 / 12.5
        id = utterance_info['ids'][0]
        flow_losses = utterance_info['flow_loss']
        token_losses = utterance_info['token_loss']

        flow_losses_timestamps = []
        if type(flow_losses) == torch.Tensor and flow_losses.ndim > 0:
            flow_losses = flow_losses.squeeze().mean(dim=-1).cpu().numpy()  # list of losses
            for idx, loss in enumerate(flow_losses):
                #print(f"Flow loss: {loss}")
                start_time = idx * REDUCTION_FACTOR * FRAMERATE
                end_time = (idx + 1) * REDUCTION_FACTOR * FRAMERATE
                flow_losses_timestamps.append((loss, start_time, end_time))
        else:
            flow_losses = None
            error_log.append(f"At file {id}, no flow_losses")

        token_losses_timestamps = []
        if type(token_losses) == torch.Tensor and token_losses.ndim > 0:
            token_losses = token_losses.squeeze().cpu().numpy()
            for idx, loss in enumerate(token_losses):
                #print(f"Token loss: {loss}")
                start_time = idx * REDUCTION_FACTOR * FRAMERATE
                end_time = (idx + 1) * REDUCTION_FACTOR * FRAMERATE
                token_losses_timestamps.append((loss, start_time, end_time))
        else:
            token_losses = None
            error_log.append(f"At file {id}, no token_losses")

        losses.append({
            'id' : id,
            'flow_losses_timestamps' : flow_losses_timestamps,
            'token_losses_timestamps' : token_losses_timestamps
        })

    with open(f"{args.root_dir}/src/gslm/tools/error_log", "a") as f:
            for i in error_log:
                f.write(i)
                f.write("\n")

    return losses

            
def process_speechocean_outputs(args, output, granularity, pooling, norm_dicts, dataset="peggy2009/speechocean_with_mfa", token=False):
    '''
    output : a list of dicts, each containing the uid, the list of flow losses, the list of token losses
    '''

    ppl_info = []
    error_log = []
    nan_count = 0
    data_main = load_dataset(dataset, split="train")
    data_main = data_main.cast_column("path", Audio(decode=False))

    data = list(data_main)

    data_by_id = {item['filename']: item for item in data}
    pbar = tqdm(output, desc=f"{granularity}-{pooling}")

    for utterance_info in pbar:
        #print(f"NEW UTTERANCE")
        if token: # word/phone level
            id = utterance_info['id']
            flow_losses_timestamps = utterance_info['flow_losses_timestamps']
            token_losses_timestamps = utterance_info['token_losses_timestamps']
            utt_data = data_by_id.get(id)

            if utt_data is None:
                error_log.append(f"File {id} is not in the hf dataset")
                continue

            # External preparation
            auc_threshold = None
            alignments = None
            human_annotation_obj = json.loads(utt_data['human_annotations'])
            human_scores = None
            phone_scores = []
            word_scores = []
            for word_obj in human_annotation_obj["words"]:
                for i in range(0, len(word_obj["phones"])):
                    phone_scores.append({
                        "phone" : word_obj["phones"][i], 
                        "accuracy" : word_obj["phones-accuracy"][i]
                    }) 
                word_scores.append({
                    "word" : word_obj["text"],
                    "accuracy" : word_obj["accuracy"],
                    "stress" : word_obj["stress"] # unused for now
                })

            # Preprocessing
            if granularity == "phone":
                # align canonical phonemes and phone alignments
                phone_alignments = json.loads(utt_data['phone_alignments'])
                phone_alignments_labels = [item['label'] for item in phone_alignments]
                phone_scores_labels = [item['phone'] for item in phone_scores]
    
                matcher = SequenceMatcher(None, phone_scores_labels, phone_alignments_labels)
                opcodes = matcher.get_opcodes()
                has_error = False
                matched_alignments = []
                matched_scores = []
                for tag, a_idx1, a_idx2, b_idx1, b_idx2 in opcodes:
                    if tag == "equal":
                        matched_scores.extend(phone_scores[a_idx1:a_idx2])
                        matched_alignments.extend(phone_alignments[b_idx1:b_idx2])
                phone_alignments = matched_alignments
                phone_scores = matched_scores
    
                if len(phone_alignments) != len(phone_scores):
                    error_log.append(f"Alignment mismatch at file {id}. {len(phone_alignments)} alignments but {len(phone_scores)} scores.")
                    error_log.append(f"{[item['label'] for item in phone_alignments]}\n{[item['phone'] for item in phone_scores]}")
    
                human_scores = phone_scores
                alignments = phone_alignments
                auc_threshold = 0.5

            elif granularity == "word":
                word_alignments = json.loads(utt_data['word_alignments'])
                if len(word_scores) != len(word_alignments):
                    raise Exception("Human word annotations cannot be aligned with word alignments.")
                human_scores = word_scores
                alignments = word_alignments
                auc_threshold = 3

            # aggregate
            if len(flow_losses_timestamps) <= 0 and len(token_losses_timestamps) <= 0:
                error_log.append(f"file {id} doesn't have valid losses")
                continue

            for loss_type in ['flow', 'token']:
                if loss_type == "flow":
                    losses_with_timestamps = flow_losses_timestamps
                    norm_dict = norm_dicts[0]
                else:
                    losses_with_timestamps = token_losses_timestamps
                    norm_dict = norm_dicts[1]

                if granularity != "utterance":
                    for i in range(0, len(alignments)):
                        current_alignment = alignments[i]

                        a_start = current_alignment["start"]
                        a_end = current_alignment["end"]
                        losses = []

                        for loss_item in losses_with_timestamps:
                            #print(loss_item)
                            t_start = loss_item[1]
                            t_end = loss_item[2]
                            if is_overlapping(a_start, a_end, t_start, t_end):
                                losses.append(loss_item[0])
                        
                        # pooling
                        loss_pooled = np.nan
                        
                        if pooling == "mean":
                            loss_pooled = np.mean(losses).item() if len(losses) > 0 else np.nan
                        elif pooling == "max":
                            loss_pooled = np.max(losses).item() if len(losses) > 0 else np.nan
                        elif pooling == "std":
                            loss_pooled = np.std(losses).item() if len(losses) > 1 else np.nan
                        else:
                            raise Exception("No pooling method specified.")
                        
                        if np.isnan(loss_pooled):
                            nan_count += 1

                        # normalization
                        if granularity == "phone":
                            # z-score normalization
                            phone_label = strip_stress(alignments[i]['label']) # type: ignore
                            p_mean = norm_dict[phone_label]['mean'] # type: ignore
                            p_std = norm_dict[phone_label]['std'] # type: ignore
                            loss_pooled_norm = ((loss_pooled - p_mean) / p_std) if p_std > 0 else np.nan
                        else :
                            word = alignments[i]['label']
                            freq = word_frequency(word, 'en')
                            neg_log_freq = -math.log(freq) if freq > 0 else np.nan  # guard against unknown words
                            w_mean = None
                            w_std = None

                            for bucket, item in norm_dict.items():
                                s = item['freq_range']
                                clean_s = s.strip("()[]")
                                left_str, right_str = clean_s.split(",")
                                left = float(left_str)
                                right = float(right_str)
                                interval = pd.Interval(left, right, closed="right")

                                if neg_log_freq in interval:
                                    w_mean = item['mean']
                                    w_std = item['std']

                            if w_mean != None and w_std != None and w_std > 0:
                                loss_pooled_norm = (loss_pooled - w_mean) / w_std
                            else:
                                loss_pooled_norm = np.nan

                        ppl_info.append({
                            "filename" : id,
                            "label" : current_alignment['label'],
                            'auc_label' : 1 if human_scores[i]['accuracy'] > auc_threshold else 0,
                            'loss_type' : loss_type,
                            "ppl_loss" : loss_pooled,
                            "ppl_loss_norm" : loss_pooled_norm,
                            "human_score": human_scores[i]['accuracy']
                        })
                else:
                    auc_threshold = 3
                    human_scores = human_annotation_obj['accuracy']

                    losses = []
                    for loss_item in losses_with_timestamps:
                        losses.append(loss_item[0])
                    
                    # pooling
                    loss_pooled = np.nan
                    
                    if pooling == "mean":
                        loss_pooled = np.mean(losses).item() if len(losses) > 0 else np.nan
                    elif pooling == "max":
                        loss_pooled = np.max(losses).item() if len(losses) > 0 else np.nan
                    elif pooling == "std":
                        loss_pooled = np.std(losses).item() if len(losses) > 1 else np.nan
                    else:
                        raise Exception("No pooling method specified.")
                    
                    if np.isnan(loss_pooled):
                        nan_count += 1
        
                    ppl_info.append({
                        "filename" : id,
                        "label" : human_annotation_obj["text"],
                        'auc_label' : 1 if human_scores > auc_threshold else 0,
                        'loss_type' : loss_type,
                        "ppl_loss" : loss_pooled,
                        "ppl_loss_norm" : np.nan,
                        "human_score": human_scores
                    })

    with open(f"{args.root_dir}/src/flow/error_log", "a") as f:
        for i in error_log:
            f.write(i)
            f.write("\n")
    return {
            "results" : ppl_info,
            "nan_count" : nan_count
        } 

def process_alignments_ds(input_dataset):
    
    alignments = []

    for sample in input_dataset:
        phone_list = []
        word_list = []

        for phone_alignment in sample['phonemes']:
            phone_list.append({
                "start" : phone_alignment['start'],
                "end" : phone_alignment["end"],
                "label" : phone_alignment["phoneme"]
            })

        for word_alignment in sample["words"]:
            word_list.append({
                "start" : word_alignment['start'],
                "end" : word_alignment["end"],
                "label" : word_alignment['word']
            })

        alignments.append({
            "audio_id" : sample['id'],
            "phone_alignment" : phone_list,
            "word_alignment" : word_list
        })
    
    return alignments

def process_librispeech_outputs(args, output, granularity, pooling, loss_type, data, alignments_ext, token=False):

    '''
    dataset         : dataset object with speaker, filename, and path
    alignments      : alignments
    pooling         : pooling method (max/mean/std)
    '''
    result_dict = {}
    error_log = []
    nan_count = 0
    data_by_id = {item['id']: item for item in data}
    alignments_by_id = {item['audio_id']: item for item in alignments_ext}

    pbar = tqdm(output, desc=f"{granularity}-{pooling}-{loss_type}")

    for utterance_info in pbar:
        # info
        if token: # word/phone level
            id = utterance_info['id']
            flow_losses_timestamps = utterance_info['flow_losses_timestamps']
            token_losses_timestamps = utterance_info['token_losses_timestamps']

            utt_data = data_by_id.get(id)
            if utt_data is None:
                error_log.append(f"File {id} is not in the hf dataset")
                continue

            # external preparation
            alignment_obj = alignments_by_id.get(id)
            phone_alignments = alignment_obj["phone_alignment"] # type: ignore # list of phone objects {start, end, label}
            word_alignments = alignment_obj["word_alignment"] # type: ignore # list of phone objects {start, end, label}

            if granularity == "phone":
                alignments = phone_alignments
            elif granularity == "word":
                alignments = word_alignments
            else:
                raise Exception("Invalid granularity.")

            # aggregate
            if len(flow_losses_timestamps) <= 0 and len(token_losses_timestamps) <= 0:
                error_log.append(f"file {id} doesn't have valid losses")
                continue

            if loss_type == "flow":
                losses_with_timestamps = flow_losses_timestamps
            else:
                losses_with_timestamps = token_losses_timestamps

            for i in range(0, len(alignments)):
                current_alignment = alignments[i]

                a_start = current_alignment["start"]
                a_end = current_alignment["end"]
                losses = []

                for loss_item in losses_with_timestamps:
                    #print(loss_item)
                    t_start = loss_item[1]
                    t_end = loss_item[2]
                    if is_overlapping(a_start, a_end, t_start, t_end):
                        losses.append(loss_item[0])
            
                # pooling
                loss_pooled = np.nan
                
                if pooling == "mean":
                    loss_pooled = np.mean(losses).item() if len(losses) > 0 else np.nan
                elif pooling == "max":
                    loss_pooled = np.max(losses).item() if len(losses) > 0 else np.nan
                elif pooling == "std":
                    loss_pooled = np.std(losses).item() if len(losses) > 1 else np.nan
                else:
                    raise Exception("No pooling method specified.")

                if granularity == "phone":
                    phone_label = strip_stress(alignments[i]['label'])

                    if phone_label in result_dict:
                        result_dict[phone_label]['count'] += 1
                        result_dict[phone_label]['losses'].append(loss_pooled)
                    else:
                        result_dict[phone_label] = {
                            "count" : 1,
                            "losses" : [loss_pooled]
                        }
                elif granularity == "word":
                    # TODO: Implement wordfreq normalization here
                    word = alignments[i]['label']
                    freq = word_frequency(word, 'en')
                    neg_log_freq = -math.log(freq) if freq > 0 else np.nan  # guard against unknown words

                    if word in result_dict:
                        result_dict[word]['freq'] = neg_log_freq
                        result_dict[word]['losses'].append(loss_pooled)
                    else:
                        result_dict[word] = {
                            'freq' : neg_log_freq,
                            'losses' : [loss_pooled]
                        }

    with open(f"{args.root_dir}/src/gslm/tools/error_log", "a") as f:
        for i in error_log:
            f.write(i)
            f.write("\n")

    return result_dict


def import_module_from_path(module_name, module_path):
    try:
        # Create a spec for the module
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        # Load the module based on the spec
        custom_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(custom_module)
        # Set the name attribute of the module
        custom_module.__name__ = module_name
        #print(f"Successfully imported module '{module_name}' from path '{module_path}'")
        return custom_module
    except Exception as e:
        print(f"Failed to import module from path '{module_path}': {e}")
        return None

def extract_number(file_path):
    # Extract the file name from the path
    file_name = file_path.split('/')[-1]
    
    # Extract the numeric part using regular expression
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group())
    else:
        return 0

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, min_lr_ratio: float = 0.1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0, min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def replace_values(original_dict, replacement_dict):
    for key, value in replacement_dict.items():
        if isinstance(value, dict):
            if key in original_dict and isinstance(original_dict[key], dict):
                replace_values(original_dict[key], value)
        else:
            if key in original_dict:
                original_dict[key] = value
