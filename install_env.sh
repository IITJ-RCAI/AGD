mamba env create -f env.yml
conda activate agd_code
python -m ipykernel install --user --name agd_code --display-name "AGD - Code ($(python --version))"
# install boost
apt-get update
apt-get install libboost-all-dev
# install maestro
git clone https://github.com/maestro-project/maestro
cd maestro
scons
cp ./maestro /usr/local/bin/maestro