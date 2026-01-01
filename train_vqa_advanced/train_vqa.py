from lightning.fabric.utilities.data import AttributeDict
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from huggingface_hub import login
from lightning.pytorch.strategies import DDPStrategy
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from pprint import pprint
from model_to_train import LightMedVLMForTraining
from utils import DataModule
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def add_callbacks():
    # Placeholder for actual callback logic (e.g., LearningRateMonitor)
    return {"callbacks": [], "loggers": []}


def train(args):

    dm = DataModule(args)
    callbacks = add_callbacks()

    if 'ddp' == args.strategy:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = args.strategy
    debug = ''   # 'debug' or ''
    # Build trainer
    if debug == 'debug':
        trainer = pl.Trainer(fast_dev_run=2)  # True (1) or number of batches
    else:
        trainer = pl.Trainer(
            devices=args.devices,
            num_nodes=args.num_nodes,
            strategy=args.strategy,
            accelerator=args.accelerator,
            precision=args.precision,
            val_check_interval=args.val_check_interval,
            limit_val_batches=args.limit_val_batches,
            limit_train_batches=args.limit_train_batches,
            limit_test_batches=args.limit_test_batches,
            max_epochs=args.max_epochs,
            num_sanity_val_steps=args.num_sanity_val_steps,
            accumulate_grad_batches=args.accumulate_grad_batches,
            callbacks=callbacks["callbacks"],
            logger=callbacks["loggers"],
            default_root_dir=args.savedmodel_path
        )

    ckpt_file = args.ckpt_file
    if ckpt_file is None:
        model = LightMedVLMForTraining(args)
    else:
        model = LightMedVLMForTraining.load_from_checkpoint(ckpt_file, strict=False, args=args)
    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)


def main(args):
    os.makedirs(args.savedmodel_path, exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)
    print('Model architecture trained successfully.')


# Allow the specific Lightning class causing the error
torch.serialization.add_safe_globals([AttributeDict])
if __name__ == '__main__':
    args = parser.parse_args(args=[])
    main(args)
