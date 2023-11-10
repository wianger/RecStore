pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118



cd dgl-0.9.1

if [[ `hostname` == "node182" ]]; then
    echo "Installing on node182"
    temp_dir="build"
else
    echo "Installing on 3090"
    temp_dir="build"
fi

mkdir ${temp_dir}
cd ${temp_dir}

cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda  ..
make -j
cd ..
cd python
python setup.py develop


cd dgl-ke
cd python
python setup.py develop
cd -

pip install paramiko ogb pyinstrument gpustat