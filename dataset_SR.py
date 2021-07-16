import gdown, pathlib

# esrgan models
esrgan_ids = {
    'RRDB_ESRGAN_x4_old_arch.pth': '1MJFgqXJrMkPdKtiuy7C6xfsU1QIbXEb-',
    'RRDB_ESRGAN_x4.pth': '1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene',
    'RRDB_PSNR_x4_old_arch.pth': '1mSJ6Z40weL-dnPvi390xDd3uZBCFMeqr',
    'RRDB_PSNR_x4.pth': '1pJ_T-V1dpb1ewoEra1TGSWl5e6H7M4NN',
}
esrgan_root = pathlib.Path('AGD_SR/search/ESRGAN/')
esrgan_root.mkdir(parents=True, exist_ok=True)
for name, id in esrgan_ids.items():
    gdown.cached_download(
        f'https://drive.google.com/uc?id={id}',
        str(esrgan_root/name)
    )