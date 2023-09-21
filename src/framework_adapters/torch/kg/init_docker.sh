cd dgl

if [[ `hostname` == "node182" ]]; then
    echo "Installing on node182"
    temp_dir="build-a30"
else
    echo "Installing on 3090"
    temp_dir="build-3090"
fi

mkdir ${temp_dir}
cd ${temp_dir}

# cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda  ..
# make -j
# cd ..
# cd python
# python setup.py develop


# cd dgl-ke
# cd python
# python setup.py develop
# cd -

# pip3 install paramiko ogb