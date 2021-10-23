# Select GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = input("Select GPU by index[15]: ") or "15"

from operations import Conv
from maestro_helpers import with_maestro
from util_gan.fid_score import compute_fid
from profile_code import profile
import pathlib

c = Conv(3, 10)
with with_maestro(True):
    print(c.forward_flops((3, 32, 32)))
with with_maestro(False):
    print(c.forward_flops((3, 32, 32)))

fid = compute_fid(
    pathlib.Path("./output/gen_epoch_1"),
    pathlib.Path("../../dataset/style_transfer/horse2zebra/test/B"),
    # use_tf='both',
)
print(f'FID: {fid}')
