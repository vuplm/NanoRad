from torch.optim.lr_scheduler import LambdaLR
import functools
from torch.optim import AdamW
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.loggers import CSVLogger
import os
import json
import re
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM, SwinModel
from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
import argparse

# ==============================================================================
# 1. USER PROVIDED DATA CLASSES (Preserved)
# ==============================================================================


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
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            def sent_cleaner(t): return re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                               .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(
                report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report

    def parse(self, features):
        to_return = {'id': features.get('image_id', features.get('id'))}

        # For VQA task: use question as input and caption as target
        if 'question' in features and 'caption' in features:
            to_return['question'] = features['question']
            # Clean the answer/caption text
            if hasattr(self.args, 'clean_answer') and self.args.clean_answer:
                text = self.clean_report(features['caption'])
            else:
                text = features['caption']
            to_return['text'] = text
        # For report generation task: use the original structure
        else:
            if self.args.task == 'label':
                text = features['label']
            elif self.args.task == 'triple':
                text = features['triple']
            elif self.args.task == 'report':
                text = features.get(
                    'report', features.get('text', ''))  # Safe get
            text = self.clean_report(text)
            to_return['text'] = text
            to_return['question'] = None  # No question for report generation

        # chest x-ray images
        images = []
        # Handle both single image_path and list of image_path
        image_paths = features.get('image_path', [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        for image_path in image_paths:
            full_path = os.path.join(self.args.base_dir, image_path)
            # Add robustness for finding images
            if not os.path.exists(full_path):
                full_path_base = os.path.join(
                    self.args.base_dir, os.path.basename(image_path))
                if os.path.exists(full_path_base):
                    full_path = full_path_base

            try:
                with Image.open(full_path) as pil:
                    array = np.array(pil, dtype=np.uint8)
                    if array.shape[-1] != 3 or len(array.shape) != 3:
                        array = np.array(pil.convert("RGB"), dtype=np.uint8)
                    image = self._parse_image(array)
                    images.append(image)
            except Exception:
                pass

        if len(images) == 0:
            images.append(torch.zeros((3, 224, 224)))

        to_return["image"] = images
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args

        # Load JSON file
        if split == 'train':
            json_file = args.annotation_train
        elif split == 'val':
            json_file = args.annotation_val
        elif split == 'test':
            json_file = args.annotation_test

        if json_file is None:
            self.meta = []
        else:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Handle both VQA format and original format
            if isinstance(data, dict) and 'annotations' in data:
                self.meta = data['annotations']
            elif isinstance(data, dict) and split in data:
                self.meta = data[split]
            elif isinstance(data, list):
                self.meta = data
            elif isinstance(data, dict):
                # Fallback for simple dict
                self.meta = list(data.values())
            else:
                self.meta = []

        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    if args.annotation_train != None:
        train_dataset = ParseDataset(args, 'train')
    else:
        train_dataset = None
    if args.annotation_val != None:
        dev_dataset = ParseDataset(args, 'val')
    else:
        dev_dataset = None  # Use train for val if null to avoid crash in setup
        if train_dataset:
            dev_dataset = train_dataset
    if args.annotation_test != None:
        test_dataset = ParseDataset(args, 'test')
    else:
        test_dataset = None
    return train_dataset, dev_dataset, test_dataset

# ==============================================================================
# 2. DATA MODULE WITH CUSTOM COLLATE
# ==============================================================================


def vqa_collate_fn(batch):
    """
    Handles list of images -> Stacked Tensor
    """
    ids = [x['id'] for x in batch]
    text = [x['text'] for x in batch]
    question = [x['question'] for x in batch]

    # Extract the first image from the list and stack
    # batch[i]['image'] is a list [Tensor(3,224,224)]
    images = torch.stack([x['image'][0] for x in batch])

    return {
        'id': ids,
        'text': text,
        'question': question,
        'image': images
    }


class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: str):
        train_dataset, dev_dataset, test_dataset = create_datasets(self.args)
        self.dataset = {
            "train": train_dataset, "validation": dev_dataset, "test": test_dataset
        }

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.args.batch_size,
                          drop_last=True, pin_memory=True, num_workers=self.args.num_workers,
                          prefetch_factor=self.args.prefetch_factor, shuffle=True,
                          collate_fn=vqa_collate_fn)  # Added collate_fn

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.args.val_batch_size,
                          drop_last=False, pin_memory=True, num_workers=self.args.num_workers,
                          prefetch_factor=self.args.prefetch_factor,
                          collate_fn=vqa_collate_fn)  # Added collate_fn

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.args.test_batch_size,
                          drop_last=False, pin_memory=False, num_workers=self.args.num_workers,
                          prefetch_factor=self.args.prefetch_factor,
                          collate_fn=vqa_collate_fn)  # Added collate_fn


# from transformers import AdamW


def add_callbacks():
    log_dir = './save/iu_xray/test'
    os.makedirs(log_dir, exist_ok=True)

    # --------- Add Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="{epoch}-{step}-{loss:.4f}",
        monitor="loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        save_weights_only=False,
        every_n_epochs=1
    )

    #     checkpoint_callback = ModelCheckpoint(
    #     dirpath=args.savedmodel_path,
    #     filename="epoch={epoch}-step={step}-val_loss={loss:.4f}",
    #     monitor="loss",
    #     mode="min",
    #     save_top_k=1,         # save best 3 checkpoints
    #     save_last=True,       # also save last.ckpt
    #     every_n_epochs=1,
    # )

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
