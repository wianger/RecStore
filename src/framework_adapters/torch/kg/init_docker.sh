cd dgl
mkdir build
cd build
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda  ..
make -j
cd ..
cd python
python setup.py develop


cd dgl-ke
cd python
python setup.py develop
cd -

pip3 install paramiko