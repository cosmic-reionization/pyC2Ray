git clone https://github.com/phirling/pyc2ray pyc2ray.git

module swap PrgEnv-cray PrgEnv-gnu
module swap gcc gcc/11.2.0

export PATH=/apps/daint/UES/hackaton/software/CUDAcore/11.8.0/bin:$PATH
module load cray-python/3.9.4.1

python3 -m venv --system-site-packages myvenv
source myvenv/bin/activate
pip install -U pip
pip install charset_normalizer numpy PyYAML tools21cm h5py # ignore error

mkdir ~/pyc2ray.git/pyc2ray/lib

cd $HOME/pyc2ray.git/src/c2ray/

make F2PY=$HOME/myvenv/bin/f2py
mv libc2ray.cpython-39-x86_64-linux-gnu.so ~/pyc2ray.git/pyc2ray/lib/

cd $HOME/pyc2ray.git/src/asora

# Makefile we set the flag --gpu-architecture=sm_60 = Nvidia P100, A100 = sm_80
grep gpu-architecture= Makefile
export PYTHONINC=-I`python -c "import sysconfig; print(sysconfig.get_path(name='include'))"`
export NUMPYINC=-I`python -c "import numpy as np; print(np.get_include())"`
make PYTHONINC=$PYTHONINC NUMPYINC=$NUMPYINC
mv libasora.so $HOME/pyc2ray.git/pyc2ray/lib/

export PYTHONPATH="$HOME/pyc2ray.git:$PYTHONPATH"
python -c "import pyc2ray as pc2r; print(' Installation succeeded')"
