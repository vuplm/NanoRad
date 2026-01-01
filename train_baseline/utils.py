import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor
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
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report


    def parse(self, features):
        to_return = {'id': features['image_id']}
        if self.args.task == 'label':
            text = features['label']
        elif self.args.task == 'triple':
            text = features['triple']
        elif self.args.task == 'report':
            text = features['caption']

        text = self.clean_report(text)
        to_return['text'] = text
        # chest x-ray images
        images = []
        for image_path in features['image_path']:
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        to_return["image"] = images
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        if split == 'train':
            self.meta = json.load(open(args.annotation_train, 'r'))
        elif split == 'val':
            self.meta = json.load(open(args.annotation_val, 'r'))
        elif split == 'test':
            self.meta = json.load(open(args.annotation_test, 'r'))
        self.meta = self.meta[split]
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
        dev_dataset = None
    if args.annotation_test != None:
        test_dataset = ParseDataset(args, 'test')
    else: 
        test_dataset = None
    return train_dataset, dev_dataset, test_dataset

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

class DataModule(LightningDataModule):

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.

        download

        tokenize

        etc…
        :return:
        """

    def setup(self, stage: str):
        """
        There are also data operations you might want to perform on every GPU. Use setup to do things like:

        count number of classes

        build vocabulary

        perform train/val/test splits

        apply transforms (defined explicitly in your datamodule or assigned in init)

        etc…
        :param stage:
        :return:
        """
        train_dataset, dev_dataset, test_dataset = create_datasets(self.args)
        self.dataset = {
            "train": train_dataset, "validation": dev_dataset, "test": test_dataset
        }


    def train_dataloader(self):
        """
        Use this method to generate the train dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=True,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor,shuffle=True)
        return loader


    def val_dataloader(self):
        """
        Use this method to generate the val dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
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
        save_top_k=1,
        every_n_train_steps=0, #args.every_n_train_steps,
        save_last=False,
        save_weights_only=False
    )
    
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, "logs"), name="tensorboard")
    csv_logger = CSVLogger(save_dir=os.path.join(log_dir, "logs"), name="csvlog")

    to_returns = {
        "callbacks": [checkpoint_callback, lr_monitor_callback],
        "loggers": [csv_logger, tb_logger]
    }
    return to_returns



def lr_lambda(current_step, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
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
    scheduler = {'scheduler': scheduler, 'name': name, 'interval': 'step', 'frequency': 1}

    return optimizer, scheduler