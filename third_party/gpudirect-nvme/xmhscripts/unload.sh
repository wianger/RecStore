# set -x
# set -e

source variable.sh

cd ${PROJECT_DIR}
# sudo rm -rf build
mkdir -p build; cd build
# cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make -j

cd ${PROJECT_DIR}/build/module
make -j
sudo make unload
