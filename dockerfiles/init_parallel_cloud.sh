cd "$(dirname "$0")"
set -x
set -e

 
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn"


sudo service ssh start

USER=xieminhui
PROJECT_PATH="/home/${USER}/RecStore"

sudo apt install -y libmemcached-dev 


ln -sf ${PROJECT_PATH}/docker_config/.bashrc ~/.bashrc
source ~/.bashrc


# git submodule add https://github.com/google/glog third_party/glog
sudo rm -f /usr/lib/x86_64-linux-gnu/libglog.so.0*

cd ${PROJECT_PATH}/third_party/glog/ && git checkout v0.5.0 && cd _build && CXXFLAGS="-fPIC" cmake .. && make -j20 && make DESTDIR=${PROJECT_PATH}/third_party/glog/glog-install-fPIC install
sudo make install


# git submodule add https://github.com/fmtlib/fmt third_party/fmt
cd ${PROJECT_PATH}/third_party/fmt/ && cd _build && CXXFLAGS="-fPIC" cmake .. && make -j20 && sudo make install


sudo apt-get update -y && \
  sudo apt-get install -y --no-install-recommends \
  libboost-all-dev \
  libevent-dev \
  libdouble-conversion-dev \
  libgflags-dev \
  libiberty-dev \
  liblz4-dev \
  liblzma-dev \
  libsnappy-dev \
  zlib1g-dev \
  binutils-dev \
  libjemalloc-dev \
  libssl-dev \
  pkg-config \
  libunwind-dev \
  libunwind8-dev \
  libelf-dev \
  libdwarf-dev \
  cloc \
  check \
  sudo \
  libtbb-dev \
  libmemcached-dev \
  libzstd-dev \
  libaio-dev

# git submodule add https://github.com/facebook/folly third_party/folly
export CC=`which gcc`
export CXX=`which g++`
cd ${PROJECT_PATH}/third_party/folly && \
# git checkout v2021.01.04.00 && \
git checkout v2023.09.11.00 && \
mkdir -p _build && cd _build \
&& CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' cmake .. -DCMAKE_INCLUDE_PATH=${PROJECT_PATH}/third_party/glog/glog-install-fPIC/usr/local/include -DCMAKE_LIBRARY_PATH=${PROJECT_PATH}/third_party/glog/glog-install-fPIC/usr/local/lib \
&& make -j20 && make DESTDIR=${PROJECT_PATH}/third_party/folly/folly-install-fPIC install

# git submodule add https://github.com/google/googletest third_party/googletest


cd ${PROJECT_PATH}/third_party/gperftools && mkdir -p _build && cd _build && CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. && make -j20 && sudo  make install

cd ${PROJECT_PATH}/third_party/cityhash/ && ./configure && make -j20 && sudo make install



cd ${PROJECT_PATH}/binary
sudo pip3 uninstall -y torch torch-tensorrt torchdata torchtext torchvision
pip3 install  -i https://pypi.tuna.tsinghua.edu.cn/simple torch-2.0.0a0+git*.whl

# GRPC
cd ${PROJECT_PATH}/
cd third_party/grpc
export MY_INSTALL_DIR=${PROJECT_PATH}/third_party/grpc-install
# rm -rf cmake/build
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
      ../..
make -j
make install -j
popd


cd ${PROJECT_PATH}/dockerfiles
# source start_core.sh


cd ${PROJECT_PATH}/src/kg/kg
bash install_dgl.sh



pip3 install pymemcache


cd /usr/lib/x86_64-linux-gnu
# sudo unlink libibverbs.so
sudo cp -f libibverbs.so.1.14.39.0  libibverbs.so
