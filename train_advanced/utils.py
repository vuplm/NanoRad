import os
import json
import re
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import zipfile
import io
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
# from transformers import AdamW
from torch.optim import AdamW
import functools
from torch.optim.lr_scheduler import LambdaLR


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(
            args.vision_model)

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(
            img, return_tensors="pt").pixel_values
        return pixel_values[0]

    def clean_report(self, report):
        if not isinstance(report, str):
            return ""

        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            def report_cleaner(t): return t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
                .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
                .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
                .strip().lower().split('. ')

            def sent_cleaner(t): return re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                               replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(
                report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            def report_cleaner(t): return t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            def sent_cleaner(t): return re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                               .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(
                report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        return report

    def parse(self, features):
        # --- FIX 1: Robust ID Extraction ---
        # Try 'id', then 'study_id', then 'subject_id', else 'unknown'
        img_id = features.get('id', features.get(
            'study_id', features.get('subject_id', 'unknown_id')))
        to_return = {'id': img_id}

        # --- FIX 2: Safe Text Extraction ---
        text = ""
        if self.args.task == 'label':
            text = features.get('label', '')
        elif self.args.task == 'triple':
            text = features.get('triple', '')
        elif self.args.task == 'report':
            # Try 'report', if missing try 'text', if missing try 'caption'
            text = features.get('report', features.get(
                'text', features.get('caption', '')))

        text = self.clean_report(text)
        to_return['text'] = text

        # --- FIX 3: Robust Image Path Logic ---
        images = []
        # Ensure image_path list exists
        image_paths = features.get('image_path', [])

        for image_path in image_paths:
            full_path = os.path.join(self.args.base_dir, image_path)

            # If path not found, try stripping "files/" or similar prefixes
            if not os.path.exists(full_path):
                parts = image_path.split('/')
                if len(parts) > 1:
                    stripped_path = os.path.join(
                        self.args.base_dir, *parts[1:])
                    if os.path.exists(stripped_path):
                        full_path = stripped_path

            try:
                with Image.open(full_path) as pil:
                    array = np.array(pil, dtype=np.uint8)
                    if array.shape[-1] != 3 or len(array.shape) != 3:
                        array = np.array(pil.convert("RGB"), dtype=np.uint8)
                    image = self._parse_image(array)
                    images.append(image)
            except Exception as e:
                # Silent fail for individual bad images, will catch below if list empty
                pass

        # Safety: If no images loaded, create a black dummy image
        if len(images) == 0:
            # print(f"Warning: No images loaded for ID {img_id}")
            images.append(torch.zeros((3, 224, 224)))

        to_return["image"] = images
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train', source_file=None):
        self.args = args
        self.parser = FieldParser(args)

        path_to_load = source_file if source_file else args.annotation
        print(f"Loading {split} dataset from: {path_to_load}")

        with open(path_to_load, 'r') as f:
            content = json.load(f)

        if isinstance(content, list):
            self.meta = content
            # DEBUG: Print keys of first item to check correctness
            if len(self.meta) > 0:
                print(
                    f"Sample keys in {split} data: {list(self.meta[0].keys())}")
        elif isinstance(content, dict):
            if split in content:
                self.meta = content[split]
            elif split == 'test' and 'val' in content:
                self.meta = content['val']
            elif split == 'val' and 'validation' in content:
                self.meta = content['validation']
            elif split == 'train' and 'training' in content:
                self.meta = content['training']
            else:
                raise KeyError(
                    f"Could not find data for '{split}' in {path_to_load}")

            if len(self.meta) > 0:
                print(
                    f"Sample keys in {split} data: {list(self.meta[0].keys())}")
        else:
            raise ValueError("JSON content must be a list or a dict.")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    # DEFINE MIMIC PATHS
    base_dir = "/kaggle/input/mimic-subset-300k"

    train_json = os.path.join(base_dir, "mimic_ready_to_train.json")
    val_json = os.path.join(base_dir, "mimic_ready_to_val.json")

    if not os.path.exists(train_json):
        print(f"Warning: {train_json} not found. Using args.annotation.")
        train_json = args.annotation
    if not os.path.exists(val_json):
        val_json = args.annotation

    train_dataset = ParseDataset(args, split='train', source_file=train_json)
    dev_dataset = ParseDataset(args, split='val', source_file=val_json)
    test_dataset = ParseDataset(args, split='val', source_file=val_json)

    return train_dataset, dev_dataset, test_dataset


class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        train_dataset, dev_dataset, test_dataset = create_datasets(self.args)
        self.dataset = {
            "train": train_dataset, "validation": dev_dataset, "test": test_dataset
        }

    def train_dataloader(self):
        loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=True,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.dataset["validation"], batch_size=self.args.val_batch_size, drop_last=False, pin_memory=True,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.dataset["test"], batch_size=self.args.test_batch_size, drop_last=False, pin_memory=False,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader


def add_callbacks():
    log_dir = './save/iu_xray/test'
    os.makedirs(log_dir, exist_ok=True)

    # --------- Add Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="{epoch}-{step}",
        save_top_k=-1,
        every_n_train_steps=0,  # args.every_n_train_steps,
        save_last=False,
        save_weights_only=False
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(log_dir, "logs"), name="tensorboard")
    csv_logger = CSVLogger(save_dir=os.path.join(
        log_dir, "logs"), name="csvlog")

    to_returns = {
        "callbacks": [checkpoint_callback, lr_monitor_callback],
        "loggers": [csv_logger, tb_logger]
    }
    return to_returns


def lr_lambda(current_step, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, float(num_training_steps - current_step) /
        float(max(1, num_training_steps - num_warmup_steps))
    )


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    return LambdaLR(optimizer, functools.partial(lr_lambda, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps), last_epoch)


def config_optimizer(parameters, init_lr, warmup_steps, max_steps, name='lr'):
    """
    Original Bert Optimizer do not decay for bias and layer_normal
    Args:
        parameters:
        init_lr:
        warmup_steps:
        max_steps:
        name:
        weight_decay:

    Returns:

    """
    optimizer = AdamW(
        parameters, lr=init_lr, eps=1e-8, correct_bias=False
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
    )
    scheduler = {'scheduler': scheduler, 'name': name,
                 'interval': 'step', 'frequency': 1}

    return optimizer, scheduler
