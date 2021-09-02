from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from torchvision.utils import save_image

import numpy as np
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

from config_eval import config
from datasets import ImageDataset, PairedImageDataset

from utils.init_func import init_weight

from utils.darts_utils import (
    create_exp_dir,
    save,
    plot_op,
    plot_path_width,
    objective_acc_lat,
)
from model_eval import NAS_GAN_Eval

from util_gan.cyclegan import Generator
from util_gan.fid_score import compute_fid
from util_gan.lr import LambdaLR

from quantize import QConv2d, QConvTranspose2d, QuantMeasure
from thop import profile
from thop.count_hooks import count_convNd

import wandb
from datetime import datetime
from matplotlib import pyplot as plt
from torchinfo import summary


def count_custom(m, x, y):
    m.total_ops += 0


custom_ops = {
    QConv2d: count_convNd,
    QConvTranspose2d: count_convNd,
    QuantMeasure: count_custom,
    nn.InstanceNorm2d: count_custom,
}


def main():
    # load env config
    config.USE_MAESTRO = os.environ.get("USE_MAESTRO", "0") == "1"
    config.TEST_RUN = os.environ.get("TEST_RUN", "0") == "1"
    config.stage = "eval"
    config.save = "ckpt/eval"
    # wandb run
    wandb.init(
        project="AGD_Maestro",
        name=f"{config.dataset}-{config.stage}-{'with' if config.USE_MAESTRO else 'without'}_maestro",
        tags=[config.dataset, "AGD", config.stage]
        + (["maestro"] if config.USE_MAESTRO else []),
        entity="rcai",
        group=os.environ.get("WANDB_GROUP", None) or f"AGD_Maestro ({datetime.now()})",
        job_type=f"Stage {config.stage}",
        reinit=True,
        sync_tensorboard=True,
        save_code=True,
        mode="disabled" if config.TEST_RUN else "online",
        config=config,
    )
    # Create logger
    logger = SummaryWriter(config.save)
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    state = torch.load(os.path.join(config.load_path, "arch.pt"))
    # Model #######################################
    model = NAS_GAN_Eval(
        state["alpha"],
        state["beta"],
        state["ratio"],
        state["beta_sh"],
        state["ratio_sh"],
        layers=config.layers,
        width_mult_list=config.width_mult_list,
        width_mult_list_sh=config.width_mult_list_sh,
        quantize=config.quantize,
    )

    if not config.real_measurement:
        flops, params = profile(
            model, inputs=(torch.randn(1, 3, 256, 256),), custom_ops=custom_ops
        )
        flops = model.forward_flops(size=(3, 256, 256))
        logger.add_scalars(
            "",
            {
                "params": params,
                "FLOPs": flops,
            },
        )
        logger.add_text("model_summary", str(summary(model, input_size=(1, 3, 256, 256))))
        print("params = %fMB, FLOPs = %fGB" % (params / 1e6, flops / 1e9))

    model = torch.nn.DataParallel(model).cuda()

    if config.ckpt:
        state_dict = torch.load(config.ckpt)
        model.load_state_dict(state_dict, strict=False)

    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    test_loader = DataLoader(
        ImageDataset(config.dataset_path, transforms_=transforms_, mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
    )

    with torch.no_grad():
        valid_fid = infer(model, test_loader, logger)
        logger.add_scalar("fid", valid_fid)
        print("Eval Fid:", valid_fid)


def infer(model, test_loader, logger):
    model.eval()

    if not config.real_measurement:
        outdir = "output/eval"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    # store image comparissions in table
    comp_table = wandb.Table(columns=["Real Image", "Generated Image"])
    once = False
    for i, batch in enumerate(test_loader):
        if not(once is True):
            once = True
            logger.add_graph(model, batch["A"].cuda())
        # Set model input
        real_A = Variable(batch["A"]).cuda()
        fake_B = 0.5 * (model(real_A).data + 1.0)

        if not config.real_measurement:
            comp_table.add_data(wandb.Image(real_A), wandb.Image(fake_B))
            save_image(fake_B, os.path.join(outdir, "%04d.png" % (i + 1)))
    wandb.log(
        {
            "Eval. Images": comp_table,
        }
    )
    if not config.real_measurement:
        fid = compute_fid(outdir, config.dataset_path + "/test/B")
        os.rename(outdir, outdir + "_%.3f" % (fid))
    else:
        fid = 0

    return fid


if __name__ == "__main__":
    main()
