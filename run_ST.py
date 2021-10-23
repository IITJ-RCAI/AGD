from datetime import datetime
import os
import pathlib
import shutil
import random

# Input config
# Select GPU
os.environ["CUDA_VISIBLE_DEVICES"] = input("Select GPU by index[15]: ") or "15"
# get metric config
if input("Should run use maestro? [y/N]: ").lower() == "y":
    os.environ["USE_MAESTRO"] = "1"
else:
    os.environ["USE_MAESTRO"] = "0"
# get run seed
main_seed = int(input("Enter the integer seed for this run[12345]: ") or "12345")
# get run repetitions
rep = int(
    input("Enter number of times to repeat run with forked random seeds[1]: ") or "1"
)

# Gen. random seeds
random.seed(main_seed)
seeds = [main_seed, *random.sample(range(int(1e7)), rep - 1)]
print(f"Random repetition seeds: {seeds}")

# Ask test run
test_run = input("Is this a test run? [y/N]: ") in ('y', 'Y')
if test_run is True:
    os.environ['TEST_RUN'] = '1'

# Set group name
dG = f"AGD_Maestro ({datetime.now()})"
os.environ["WANDB_GROUP"] = input(f"WanDB Group name[{dG}]: ") or dG

# Ask for backup
# bkup = input("Do you want to backup ckpt/<stage> folder data? [y/N]: ") in ('y', 'Y')
bkup = False


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


def clean_slate():
    # clear ckpt and output dirs
    rmdirs = [
        pathlib.Path("./AGD_ST/search/ckpt/*"),
        pathlib.Path("./AGD_ST/search/output/*"),
        pathlib.Path("./AGD_ST/search/*_lookup_table.npy"),
    ]
    for r in rmdirs:
        os.system(f"rm -rf {str(r)}")


# get input for clean slate run
csl = input("Is this a clean slate run?[y/N]: ")
csl = True if csl == "y" or csl == "Y" else False
if csl is True:
    ni = True
    ni_str = "yyy"
else:
    # get input for non-interactive
    if rep > 1:
        print("More than 1 repetitions, treating this as a non-interactive run.")
        ni = True
    else:
        ni = input("Is this a non-interactive run?[y/N]: ")
        ni = True if ni == "y" or ni == "Y" else False
    if ni:
        ni_str = input(
            "Please provide y/n string for the next 3 runs [pretrain/train/finetune]: "
        )
        if len(ni_str) != 3:
            print("Invalid input.")
            exit()

if ni:
    banner("Running in Non-Interactive mode")

# prepare dataset
print("Preparing datasets...")
os.system("python dataset_ST.py")
print("Done")

task_st = pathlib.Path("./AGD_ST")
ckpt = task_st / "search" / "ckpt"


for seed_idx, rng_seed in enumerate(seeds):
    run_prefix = f"seed{rng_seed}"
    print(f"Running repetetion {seed_idx+1} with rng seed: {rng_seed}")
    random.seed(rng_seed)
    os.environ['RNG_SEED'] = str(rng_seed)
    if csl is True:
        clean_slate()
    # run pre-train
    banner("Pre-train")
    pre_ckpt = ckpt / "pretrain"
    skip = False
    if pre_ckpt.exists():
        ip = (
            input("Pre-train checkpoint exists. Want to overrite?[Y/n]: ")
            if not ni
            else ni_str[0]
        )
        if ip == "Y" or ip == "y" or ip == "":
            shutil.rmtree(pre_ckpt, ignore_errors=True)
        else:
            skip = True
    if not skip:
        print("Running pre-train...")
        os.system("cd AGD_ST/search && python train_search.py")
        # backup checkpoint
        if bkup:
            tar_file = pathlib.Path(f"{run_prefix}_pretrain_ckpt.tar.gz")
            if tar_file.exists():
                nn = number_this_file(tar_file)
                tar_file.rename(nn)
            os.system(f"tar -czvf {str(tar_file)} -C {str(ckpt)} {str(pre_ckpt.name)}")
        print("Done")

    # run train_search
    banner("Train search")
    skip = False
    train_ckpt = ckpt / "search"
    if not pre_ckpt.exists():
        print("Please pre-train before training.")
        exit()
    if train_ckpt.exists():
        ip = (
            input("Train checkpoint exists. Want to overrite?[Y/n]: ")
            if not ni
            else ni_str[1]
        )
        if ip == "Y" or ip == "y" or ip == "":
            shutil.rmtree(train_ckpt, ignore_errors=True)
        else:
            skip = True
    if not skip:
        print("Running train search...")
        # rename config files
        cfg_train = task_st / "search" / "config_search.py.train"
        cfg_pre = task_st / "search" / "config_search.py"
        # swap configs
        cfg_pre.rename(cfg_pre.parent / "config_search.py.pretrain")
        cfg_pre = cfg_pre.parent / "config_search.py.pretrain"
        cfg_train.rename(cfg_train.parent / "config_search.py")
        cfg_train = cfg_train.parent / "config_search.py"
        try:
            os.system("cd AGD_ST/search && python train_search.py")
            # backup checkpoint
            if bkup:
                tar_file = pathlib.Path(f"{run_prefix}_train_ckpt.tar.gz")
                if tar_file.exists():
                    nn = number_this_file(tar_file)
                    tar_file.rename(nn)
                os.system(f"tar -czvf {str(tar_file)} -C {str(ckpt)}h {str(train_ckpt.name)}")
        finally:
            # Replace normal config files
            cfg_train.rename(cfg_train.parent / "config_search.py.train")
            cfg_pre.rename(cfg_pre.parent / "config_search.py")
        print("Done")

    # Train from scratch
    banner("Train from scratch")
    train_sc_ckpt = ckpt / "finetune"
    skip = False
    if train_sc_ckpt.exists():
        ip = (
            input("Finetune train checkpoint exists. Want to overrite?[Y/n]: ")
            if not ni
            else ni_str[2]
        )
        if ip == "Y" or ip == "y" or ip == "":
            shutil.rmtree(train_sc_ckpt, ignore_errors=True)
        else:
            skip = True
    if not skip:
        print("Running finetune...")
        os.system("cd AGD_ST/search && python train.py")
        # backup checkpoint
        if bkup:
            tar_file = pathlib.Path(f"{run_prefix}_finetune_ckpt.tar.gz")
            if tar_file.exists():
                nn = number_this_file(tar_file)
                tar_file.rename(nn)
            os.system(f"tar -czvf {str(tar_file)} -C {str(ckpt)} {str(train_sc_ckpt.name)}")
        print("Done")

    # Eval
    banner("Evaluate")
    print("Evaluating trained model...")
    os.system("cd AGD_ST/search && python eval.py")
    print("Done")
