module purge
module load Stages/2025
module load GCC/13.3.0
module load Python/3.12.3
module load CUDA
module load ParaStationMPI
module load MPI-settings/CUDA

source /e/project1/lgreion/david/bin/pyc2rayLW-env/bin/activate

cd /e/project1/lgreion/david/LW_project/pyC2Ray

#!/bin/bash
set -e

ENV=/e/project1/lgreion/david/bin/pyc2rayLW-env
REPO=/e/project1/lgreion/david/LW_project/pyC2Ray

source "$ENV/bin/activate"
cd "$REPO"

echo "=== Python environment ==="
which python
python -m pip --version

echo
echo "=== Building PyC2Ray for GH200 / sm_90 ==="

python -m pip install . \
    --no-build-isolation \
    --config-settings=setup-args="-Dgpu-architecture=90"

echo
echo "=== Copying sm_90 ASORA into live source tree ==="

cp \
"$ENV/lib/python3.12/site-packages/pyc2ray/lib/libasora.cpython-312-aarch64-linux-gnu.so" \
"$REPO/pyc2ray/lib/libasora.so"

echo
echo "=== Checking imported PyC2Ray ==="

python - <<'PY'
import pyc2ray
import pyc2ray.lib.libasora as asora

print("PyC2Ray:", pyc2ray.__file__)
print("ASORA:", asora.__file__)
PY

ASORA=$(python -c "import pyc2ray.lib.libasora as a; print(a.__file__)")

echo
echo "=== CUDA architecture ==="
cuobjdump --list-elf "$ASORA"

echo
echo "=== DONE ==="