import torch
import torch.utils.data.dataloader
import importlib
import warnings

from collections.abc import Mapping, Sequence, Iterable

from lib.utils import TensorDict, TensorList

warnings.filterwarnings("ignore")

# =========================================================
# FIX: compatibility for old ARTrack code
# =========================================================

string_classes = (str, bytes)
int_classes = int


# =========================================================
# Shared memory helper
# =========================================================

def _check_use_shared_memory():
    if hasattr(torch.utils.data.dataloader, '_use_shared_memory'):
        return getattr(torch.utils.data.dataloader, '_use_shared_memory')

    collate_lib = importlib.import_module('torch.utils.data._utils.collate')

    if hasattr(collate_lib, '_use_shared_memory'):
        return getattr(collate_lib, '_use_shared_memory')

    return torch.utils.data.get_worker_info() is not None


# =========================================================
# COLLATE FUNCTION (dim 0)
# =========================================================

def ltr_collate(batch):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"

    elem_type = type(batch[0])

    # ---------------- Tensor case ----------------
    if isinstance(batch[0], torch.Tensor):
        out = None

        if _check_use_shared_memory():
            numel = sum(x.numel() for x in batch)
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)

        return torch.stack(batch, 0, out=out)

    # ---------------- numpy case ----------------
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ not in ['str_', 'string_']:
        elem = batch[0]

        if elem_type.__name__ == 'ndarray':
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)

        if elem.shape == ():
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](
                list(map(py_type, batch))
            )

    # ---------------- primitive types ----------------
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch

    # ---------------- custom types ----------------
    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate([d[key] for d in batch]) for key in batch[0]})

    elif isinstance(batch[0], Mapping):
        return {key: ltr_collate([d[key] for d in batch]) for key in batch[0]}

    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate(samples) for samples in transposed])

    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [ltr_collate(samples) for samples in transposed]

    elif batch[0] is None:
        return batch

    raise TypeError(error_msg.format(type(batch[0])))


# =========================================================
# COLLATE FUNCTION (dim 1)
# =========================================================

def ltr_collate_stack1(batch):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"

    elem_type = type(batch[0])

    if isinstance(batch[0], torch.Tensor):
        out = None

        if _check_use_shared_memory():
            numel = sum(x.numel() for x in batch)
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)

        return torch.stack(batch, 1, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ not in ['str_', 'string_']:
        elem = batch[0]

        if elem_type.__name__ == 'ndarray':
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 1)

        if elem.shape == ():
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](
                list(map(py_type, batch))
            )

    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch

    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]})

    elif isinstance(batch[0], Mapping):
        return {key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]}

    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate_stack1(samples) for samples in transposed])

    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [ltr_collate_stack1(samples) for samples in transposed]

    elif batch[0] is None:
        return batch

    raise TypeError(error_msg.format(type(batch[0])))


# =========================================================
# DATA LOADER
# =========================================================

class LTRLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(
        self,
        name,
        dataset,
        training=True,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        epoch_interval=1,
        collate_fn=None,
        stack_dim=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None
    ):

        print("pin_memory is", pin_memory)

        if collate_fn is None:
            if stack_dim == 0:
                collate_fn = ltr_collate
            elif stack_dim == 1:
                collate_fn = ltr_collate_stack1
            else:
                raise ValueError("Stack dim not supported. Must be 0 or 1.")

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn
        )

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim
