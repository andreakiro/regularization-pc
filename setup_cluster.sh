. /cluster/apps/local/env2lmod.sh
module load gcc/6.3.0
module load hdf5/1.10.1
module load python/3.8.5
module load eth_proxy

python -m venv "${SCRATCH}/.venv"
. "${SCRATCH}/.venv/bin/activate"

python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install -r ./env/requirements.txt
