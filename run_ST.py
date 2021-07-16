import os, pathlib, shutil

# io func
def banner(msg):
    print('='*20)
    print(msg)
    print('='*20)


# get input for non-interactive
ni = input(f'Is this a non-interactive run?[y/N]: ')
ni = True if ni=='y' or ni=='Y' else False

if ni:
    banner(f'Running in Non-Interactive mode')

# prepare dataset
print(f'Preparing datasets...')
os.system(f'python dataset.py')
print(f'Done')

task_st = pathlib.Path('./AGD_ST')
ckpt = task_st / 'search' / 'ckpt'

# run pre-train
banner('Pre-train')
pre_ckpt = ckpt / 'pretrain'
skip = False
if pre_ckpt.exists():
    ip = input(f'Pre-train checkpoint exists. Want to overrite?[Y/n]: ') if not ni else 'y'
    if ip == 'Y' or ip =='y' or ip == '':
        shutil.rmtree(pre_ckpt, ignore_errors=True)
    else:
        skip = True
if not skip:
    print(f'Running pre-train...')
    os.system(f'cd AGD_ST/search && python train_search.py')
    print(f'Done')

# run train_search
banner('Train search')
skip = False
train_ckpt = task_st / 'search' / 'ckpt' / 'search'
if not pre_ckpt.exists():
    print(f'Please pre-train before training.')
    exit()
if train_ckpt.exists():
    ip = input(f'Train checkpoint exists. Want to overrite?[Y/n]: ') if not ni else 'y'
    if ip == 'Y' or ip =='y' or ip == '':
        shutil.rmtree(train_ckpt, ignore_errors=True)
    else:
        skip = True
if not skip:
    print(f'Running train search...')
    # rename config files
    cfg_train = task_st / 'search' / 'config_search.py.train'
    cfg_pre = task_st / 'search' / 'config_search.py'
    # swap configs
    cfg_pre.rename(cfg_pre.parent / 'config_search.py.pretrain')
    cfg_pre = cfg_pre.parent / 'config_search.py.pretrain'
    cfg_train.rename(cfg_train.parent / 'config_search.py')
    cfg_train = cfg_train.parent / 'config_search.py'
    try:
        os.system(f'cd AGD_ST/search && python train_search.py')
    finally:
        # Replace normal config files
        cfg_train.rename(cfg_train.parent / 'config_search.py.train')
        cfg_pre.rename(cfg_pre.parent / 'config_search.py')
    print(f'Done')

# Train from scratch
banner(f'Train from scratch')
train_sc_ckpt = ckpt / 'finetune'
skip = False
if train_sc_ckpt.exists():
    ip = input(f'Finetune train checkpoint exists. Want to overrite?[Y/n]: ') if not ni else 'y'
    if ip == 'Y' or ip =='y' or ip == '':
        shutil.rmtree(train_sc_ckpt, ignore_errors=True)
    else:
        skip = True
if not skip:
    print(f'Running finetune...')
    os.system(f'cd AGD_ST/search && python train.py')
    print(f'Done')

# Eval
banner(f'Evaluvate')
print(f'Evaluating trained model...')
os.system(f'cd AGD_ST/search && python eval.py')
print(f'Done')