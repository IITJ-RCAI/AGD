from datetime import datetime
import os
import pathlib
import shutil
import wandb


# io func
def banner(msg):
    print("=" * 20)
    print(msg)
    print("=" * 20)


def number_this_file(fl: pathlib.Path):
    i = 0
    while fl.exists():
        i += 1
        fl = fl.parent / (fl.name + f".{i}")
    return fl


# Select GPU
os.environ["CUDA_VISIBLE_DEVICES"] = input("Select GPU by index[15]: ") or "15"

# get metric config
if input("Should run use maestro? [y/N]: ").lower() == "y":
    os.environ["USE_MAESTRO"] = "1"
else:
    os.environ["USE_MAESTRO"] = "0"

# Set group name
dG = f"AGD_Maestro ({datetime.now()})"
os.environ["WANDB_GROUP"] = input(f"WanDB Group name[{dG}]: ") or dG

# get input for non-interactive
ni = input(f"Is this a non-interactive run?[y/N]: ")
ni = True if ni == "y" or ni == "Y" else False
if ni:
    ni_str = input(
        f"Please provide y/n string for the next 3 runs [pretrain/train/finetune]: "
    )
    if len(ni_str) != 3:
        print(f"Invalid input.")
        exit()

if ni:
    banner(f"Running in Non-Interactive mode")

# prepare dataset
print(f"Preparing datasets...")
os.system(f"python dataset_ST.py")
print(f"Done")

task_st = pathlib.Path("./AGD_ST")
ckpt = task_st / "search" / "ckpt"

# run pre-train
banner("Pre-train")
pre_ckpt = ckpt / "pretrain"
skip = False
if pre_ckpt.exists():
    ip = (
        input(f"Pre-train checkpoint exists. Want to overrite?[Y/n]: ")
        if not ni
        else ni_str[0]
    )
    if ip == "Y" or ip == "y" or ip == "":
        shutil.rmtree(pre_ckpt, ignore_errors=True)
    else:
        skip = True
if not skip:
    print(f"Running pre-train...")
    os.system(f"cd AGD_ST/search && python train_search.py")
    # compress checkpoint
    tar_file = pathlib.Path("pretrain_ckpt.tar.gz")
    if tar_file.exists():
        nn = number_this_file(tar_file)
        tar_file.rename(nn)
    os.system(f"tar -czvf {str(tar_file)}  -C AGD_ST/search {str(pre_ckpt)}")
    print(f"Done")

# run train_search
banner("Train search")
skip = False
train_ckpt = ckpt / "search"
if not pre_ckpt.exists():
    print(f"Please pre-train before training.")
    exit()
if train_ckpt.exists():
    ip = (
        input(f"Train checkpoint exists. Want to overrite?[Y/n]: ")
        if not ni
        else ni_str[1]
    )
    if ip == "Y" or ip == "y" or ip == "":
        shutil.rmtree(train_ckpt, ignore_errors=True)
    else:
        skip = True
if not skip:
    print(f"Running train search...")
    # rename config files
    cfg_train = task_st / "search" / "config_search.py.train"
    cfg_pre = task_st / "search" / "config_search.py"
    # swap configs
    cfg_pre.rename(cfg_pre.parent / "config_search.py.pretrain")
    cfg_pre = cfg_pre.parent / "config_search.py.pretrain"
    cfg_train.rename(cfg_train.parent / "config_search.py")
    cfg_train = cfg_train.parent / "config_search.py"
    try:
        os.system(f"cd AGD_ST/search && python train_search.py")
        # compress checkpoint
        tar_file = pathlib.Path("train_ckpt.tar.gz")
        if tar_file.exists():
            nn = number_this_file(tar_file)
            tar_file.rename(nn)
        os.system(f"tar -czvf {str(tar_file)}  -C AGD_ST/search {str(train_ckpt)}")
    finally:
        # Replace normal config files
        cfg_train.rename(cfg_train.parent / "config_search.py.train")
        cfg_pre.rename(cfg_pre.parent / "config_search.py")
    print(f"Done")

# Train from scratch
banner(f"Train from scratch")
train_sc_ckpt = ckpt / "finetune"
skip = False
if train_sc_ckpt.exists():
    ip = (
        input(f"Finetune train checkpoint exists. Want to overrite?[Y/n]: ")
        if not ni
        else ni_str[2]
    )
    if ip == "Y" or ip == "y" or ip == "":
        shutil.rmtree(train_sc_ckpt, ignore_errors=True)
    else:
        skip = True
if not skip:
    print(f"Running finetune...")
    os.system(f"cd AGD_ST/search && python train.py")
    # compress checkpoint
    tar_file = pathlib.Path("finetune_ckpt.tar.gz")
    if tar_file.exists():
        nn = number_this_file(tar_file)
        tar_file.rename(nn)
    os.system(f"tar -czvf {str(tar_file)}  -C AGD_ST/search {str(train_sc_ckpt)}")
    print(f"Done")

# Eval
banner(f"Evaluvate")
print(f"Evaluating trained model...")
os.system(f"cd AGD_ST/search && python eval.py")
print(f"Done")
