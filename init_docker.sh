export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
set -x
set -e
# git config --global user.name "Minhui Xie"
# git config --global user.email "645214784@qq.com"

sudo service ssh start


PROJECT_PATH="/home/${USER}/RecStore"


sudo apt install -y libmemcached-dev 


ln -sf ${PROJECT_PATH}/docker_config/.bashrc /home/${USER}/.bashrc
source /home/${USER}/.bashrc


# git submodule add https://github.com/google/glog third_party/glog
sudo rm -f /usr/lib/x86_64-linux-gnu/libglog.so.0*
cd ${PROJECT_PATH}/third_party/glog/ && git checkout v0.5.0 && rm -rf _build && mkdir _build && cd _build && CXXFLAGS="-fPIC" cmake .. && make -j20 && sudo make install


# git submodule add https://github.com/fmtlib/fmt third_party/fmt
cd ${PROJECT_PATH}/third_party/fmt/ && rm -rf _build && mkdir _build && cd _build && CXXFLAGS="-fPIC" cmake .. && make -j20 && sudo make install


# git submodule add https://github.com/facebook/folly third_party/folly
cd ${PROJECT_PATH}/third_party/folly && git checkout v2021.01.04.00 && rm -rf _build && mkdir -p _build && cd _build && CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. && make -j20 && make DESTDIR=${PROJECT_PATH}/third_party/folly/folly-install-fPIC install && make clean

# git submodule add https://github.com/google/googletest third_party/googletest


cd ${PROJECT_PATH}/third_party/gperftools && rm -rf _build && mkdir -p _build && cd _build && CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. && make -j20 && sudo  make install && make clean


# cd ${PROJECT_PATH}/third_party/gperftools/ && ./autogen.sh && ./configure && make -j20 && sudo make install

cd ${PROJECT_PATH}/third_party/cityhash/ && ./configure && make -j20 && sudo make install

# cd ${PROJECT_PATH}/third_party/rocksdb/ && rm -rf _build && mkdir _build && cd _build && cmake .. && make -j20 && sudo make install

#############################SPDK#############################
cd ${PROJECT_PATH}/

sudo apt install -y ca-certificates

# sudo cp docker_config/ubuntu20.04.apt.ustc /etc/apt/sources.list
sudo sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
sudo -E apt-get update

cd third_party/spdk
sudo PATH=$PATH which pip3

# if failed, sudo su, and execute in root;
# the key is that which pip3 == /opt/bin/pip3
sudo -E PATH=$PATH scripts/pkgdep.sh --all
# exit sudo su

./configure
sudo make clean
make -j20
sudo make install
# make clean
#############################SPDK#############################
sudo rm /opt/conda/lib/libtinfo.so.6


# GRPC
cd ${PROJECT_PATH}/
cd third_party/grpc
export MY_INSTALL_DIR=${PROJECT_PATH}/third_party/grpc-install
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
      ../..
make -j20
sudo make install -j
popd

sudo apt install -y sshpass
ssh-keygen -t rsa -q -f "$HOME/.ssh/id_rsa" -N ""
sshpass  -p 1234 ssh-copy-id 10.0.2.182


mkdir -p ~/.config/dask
echo "distributed:
  worker:
    # Fractions of worker memory at which we take action to avoid memory blowup
    # Set any of the lower three values to False to turn off the behavior entirely
    memory:
      target: 0.60  # target fraction to stay below
      spill: 0.70  # fraction at which we spill to disk
      pause: 0.80  # fraction at which we pause worker threads
      terminate: False  # fraction at which we terminate the worker" >~/.config/dask/distributed.yaml
