import gdown, pathlib
from dataset.super_resolution.extract_images import main as extract_images

# esrgan models
esrgan_ids = {
    "RRDB_ESRGAN_x4_old_arch.pth": "1MJFgqXJrMkPdKtiuy7C6xfsU1QIbXEb-",
    "RRDB_ESRGAN_x4.pth": "1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene",
    "RRDB_PSNR_x4_old_arch.pth": "1mSJ6Z40weL-dnPvi390xDd3uZBCFMeqr",
    "RRDB_PSNR_x4.pth": "1pJ_T-V1dpb1ewoEra1TGSWl5e6H7M4NN",
}
esrgan_root = pathlib.Path("AGD_SR/search/ESRGAN/")
esrgan_root.mkdir(parents=True, exist_ok=True)
for name, id in esrgan_ids.items():
    gdown.cached_download(
        f"https://drive.google.com/uc?id={id}", str(esrgan_root / name)
    )

# Div2K dataset
links = [
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X2.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X2.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X3.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X3.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X4.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_x8.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_mild.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_difficult.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_wild.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_x8.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_mild.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_difficult.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_wild.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
]
data_root = pathlib.Path("dataset/super_resolution/div2k")
for l in links:
    gdown.cached_download(
        l,
        str(data_root / l.split("/")[-1]),
        postprocess=gdown.extractall,
    )
extract_images(data_root)
# Download eval data
gdown.cached_download(
    'http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip',
    str(data_root / 'eval' / 'SR_eval.zip'),
    postprocess=gdown.extractall
)
