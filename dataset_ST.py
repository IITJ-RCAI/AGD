import gdown, pathlib

dr = pathlib.Path('dataset')
dr.mkdir(exist_ok=True, parents=True)

style_tr = dr / 'style_transfer.tar'

# style transfer
gdown.cached_download(
    'https://drive.google.com/uc?id=1HeL4YGtXF22nyIN3bDLI5WbhYhTYVYiv',
    str(style_tr),
    postprocess=gdown.extractall
)