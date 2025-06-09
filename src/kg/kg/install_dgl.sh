set -e
set -x

# sudo apt-get install libboost-all-dev

cd dgl-0.9.1
git checkout 0.9.1

if [[ `hostname` == "node182" ]]; then
    echo "Installing on node182"
    temp_dir="build"
else
    echo "Installing on 3090"
    temp_dir="build"
fi

rm -rf ${temp_dir}
mkdir ${temp_dir}
cd ${temp_dir}

# export CC=`which gcc-7`
# export CXX=`which g++-7`
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..
make -j
cd ..
cd python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy==1.12.0rc1 
python setup.py develop --user


cd ../..
cd dgl-ke
cd python
python setup.py develop --user
cd -

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple paramiko ogb pyinstrument gpustat debugpy pytest
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "pybind11[global]"


# conda install fmt
