import argparse
import logging
import os
import pathlib
from typing import List, NoReturn
import pytorch_lightning as pl
# from pytorch_lightning.strategies import DDPStrategy
from torch.utils.tensorboard import SummaryWriter
from data.datamodules import *
from utils import create_logging, parse_yaml
from models.resunet import *
from losses import get_loss_function
from models.audiosep import AudioSep, get_model_class
from data.waveform_mixers import SegmentMixer
from models.clap_encoder import CLAP_Encoder
from callbacks.base import CheckpointEveryNSteps
from optimizers.lr_schedulers import get_lr_lambda
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


def get_dirs(
    workspace: str, 
    filename: str, 
    config_yaml: str, 
    devices_num: int
) -> List[str]:
    r"""Get directories and paths.

    Args:
        workspace (str): directory of workspace
        filename (str): filename of current .py file.
        config_yaml (str): config yaml path
        devices_num (int): 0 for cpu and 8 for training with 8 GPUs

    Returns:
        checkpoints_dir (str): directory to save checkpoints
        logs_dir (str), directory to save logs
        tf_logs_dir (str), directory to save TensorBoard logs
        statistics_path (str), directory to save statistics
    """
    
    os.makedirs(workspace, exist_ok=True)

    yaml_name = pathlib.Path(config_yaml).stem

    # Directory to save checkpoints
    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        filename,
        #"{},devices={}".format(yaml_name, devices_num),
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Directory to save logs
    logs_dir = os.path.join(
        workspace,
        "logs",
        filename,
        #"{},devices={}".format(yaml_name, devices_num),
    )
    os.makedirs(logs_dir, exist_ok=True)

    # Directory to save TensorBoard logs
    create_logging(logs_dir, filemode="w")
    logging.info(args)

    tf_logs_dir = os.path.join(
        workspace,
        "tf_logs",
        filename,
        #"{},devices={}".format(yaml_name, devices_num),
    )

    # Directory to save statistics
    statistics_path = os.path.join(
        workspace,
        "statistics",
        filename,
        #"{},devices={}".format(yaml_name, devices_num),
        "statistics.pkl",
    )
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)

    return checkpoints_dir, logs_dir, tf_logs_dir, statistics_path

 
def get_data_module(
    configs,
    num_workers: int,
    batch_size: int,
) -> DataModule:
    r"""Create data_module. Mini-batch data can be obtained by:

    code-block:: python

        data_module.setup()

        for batch_data_dict in data_module.train_dataloader():
            print(batch_data_dict.keys())
            break

    Args:
        workspace: str
        config_yaml: str
        num_workers: int, e.g., 0 for non-parallel and 8 for using cpu cores
            for preparing data in parallel
        distributed: bool

    Returns:
        data_module: DataModule
    """

    sampling_rate = configs['data']['sampling_rate']
    segment_seconds = configs['data']['segment_seconds']
    
    # audio-text datasets
    print('Getting datafiles ...')
    datafiles = configs['data']['datafiles']
    
    datafiles_valid = configs['data']['datafiles_valid']
    print('Getting datafiles down.')
    # dataset
    dataset = AudioTextDataset(
        datafiles=datafiles, 
        sampling_rate=sampling_rate, 
        max_clip_len=segment_seconds,
    )
    
    dataset_valid = AudioTextDataset(
        datafiles=datafiles_valid, 
        sampling_rate=sampling_rate, 
        max_clip_len=segment_seconds,
    )
    # data module
    data_module = DataModule(
        train_dataset=dataset,
        val_dataset=dataset_valid,
        num_workers=num_workers,
        batch_size=batch_size
    )

    return data_module


def train(args) -> NoReturn:
    r"""Train, evaluate, and save checkpoints.

    Args:
        workspace: str, directory of workspace
        gpus: int, number of GPUs to train
        config_yaml: str
    """
    # arguments & parameters
    workspace = args.workspace
    config_yaml = args.config_yaml
    filename = args.filename

    devices_num = torch.cuda.device_count()
    
    # Read config file.
    print('Getting config yaml ...')
    configs = parse_yaml(config_yaml)
    print('Getting config yaml down')
    # Configuration of data
    max_mix_num = configs['data']['max_mix_num']
    sampling_rate = configs['data']['sampling_rate']
    lower_db = configs['data']['loudness_norm']['lower_db']
    higher_db = configs['data']['loudness_norm']['higher_db']

    # Configuration of the separation model
    query_net = configs['model']['query_net']
    model_type = configs['model']['model_type']
    input_channels = configs['model']['input_channels']
    output_channels = configs['model']['output_channels']
    condition_size = configs['model']['condition_size']
    use_text_ratio = configs['model']['use_text_ratio']
    dprnn = configs['model']['dprnn']
    dprnn_layers = configs['model']['dprnn_layers']
    dprnn_hidden = configs['model']['dprnn_hidden']
    
    # Configuration of the trainer
    num_nodes = configs['train']['num_nodes']
    batch_size = configs['train']['batch_size_per_device'] 
    sync_batchnorm = configs['train']['sync_batchnorm'] 
    num_workers = configs['train']['num_workers']
    loss_type = configs['train']['loss_type']
    optimizer_type = configs["train"]["optimizer"]["optimizer_type"]
    learning_rate = float(configs['train']["optimizer"]['learning_rate'])
    lr_lambda_type = configs['train']["optimizer"]['lr_lambda_type']
    warm_up_steps = configs['train']["optimizer"]['warm_up_steps']
    reduce_lr_steps = configs['train']["optimizer"]['reduce_lr_steps']
    save_step_frequency = configs['train']['save_step_frequency']
    freeze = configs['train']['freeze']
    accumulate_grad = configs['train']['accumulation_grad']
    load_pretain = configs['train']['load_pretrain']
    train_max_epoch = configs['train']['train_max_epoch']
    
    
    resume_checkpoint_path = args.resume_checkpoint_path
    if resume_checkpoint_path == "":
        resume_checkpoint_path = None
    else:
        logging.info(f'Finetuning AudioSep with checkpoint [{resume_checkpoint_path}]')

    # Get directories and paths
    checkpoints_dir, logs_dir, tf_logs_dir, statistics_path = get_dirs(
        workspace, filename, config_yaml, devices_num,
    )

    logging.info(configs)

    # data module
    print('Geting data module ...')
    
    data_module = get_data_module(
        configs=configs,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    print('Geting data module down.')
    
    # model
    Model = get_model_class(model_type=model_type)

    ss_model = Model(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
        dprnn=dprnn,
        dprnn_layers=dprnn_layers,
        dprnn_hidden=dprnn_hidden
    )

    # loss function
    loss_function = get_loss_function(loss_type)

    segment_mixer = SegmentMixer(
        max_mix_num=max_mix_num,
        lower_db=lower_db, 
        higher_db=higher_db
    )

    
    if query_net == 'CLAP':
        query_encoder = CLAP_Encoder()
    else:
        raise NotImplementedError

    lr_lambda_func = get_lr_lambda(
        lr_lambda_type=lr_lambda_type,
        warm_up_steps=warm_up_steps,
        reduce_lr_steps=reduce_lr_steps,
    )

    # pytorch-lightning model
    pl_model = AudioSep(
        ss_model=ss_model,
        waveform_mixer=segment_mixer,
        query_encoder=query_encoder,
        loss_function=loss_function,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        lr_lambda_func=lr_lambda_func,
        use_text_ratio=use_text_ratio,
        freeze = freeze,
        batchsize=batch_size,
    )
    
    # baseline:
    # checkpoint_every_n_steps = CheckpointEveryNSteps(
    #     checkpoints_dir=checkpoints_dir,
    #     save_step_frequency=save_step_frequency,
    # )
    
    # callbacks = [checkpoint_every_n_steps]
    
    checkpoint_callback = ModelCheckpoint(monitor=('val_sdr'), 
                                          mode='max',
                                          save_top_k=3,
                                          dirpath=checkpoints_dir,
                                          save_weights_only=False,
                                          filename='model-{epoch:02d}-{val_sdr:.4f}')
    
    callbacks = [checkpoint_callback]

    summary_writer = SummaryWriter(log_dir=tf_logs_dir)

    trainer = pl.Trainer(
        accelerator='auto',
        devices=[0,1], # 'auto'
        strategy='ddp', # ddp_find_unused_parameters_true
        num_nodes=num_nodes,
        precision=32, # 32-true
        logger=None,
        callbacks=callbacks,
        fast_dev_run=False,
        max_epochs=train_max_epoch,
        log_every_n_steps=50,
        # use_distributed_sampler=True,
        sync_batchnorm=sync_batchnorm,
        num_sanity_val_steps=2,
        enable_checkpointing=True, # base: False
        enable_progress_bar=True,
        enable_model_summary=True,
        accumulate_grad_batches=accumulate_grad,
        
    )
    
    if load_pretain:
        ## ================= load pretraind model ======================== ## 
        checkpoint_path = resume_checkpoint_path
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        pl_model.load_state_dict(checkpoint_data['state_dict'], strict=True)
        # ================================================================= #

    # Fit, evaluate, and save checkpoints.
    trainer.fit(
        model=pl_model, 
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=data_module, # data_module
        ckpt_path=None,  # resume_checkpoint_path
    )
    

if __name__ == "__main__":
    print('=== INIT DOWN ===')
    # set GPUs
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )

    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        required=True,
        default='',
        help="Path of pretrained checkpoint for finetuning.",
    )

    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        default='',
        help="filename of current workspace.",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        required=False,
        default='',
        help="GPU used.",
    )

    args = parser.parse_args()
    # args.filename = pathlib.Path(__file__).stem
    
    train(args)