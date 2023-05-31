# set -x
set -e

echo execute in docker, dont in host
source variable.sh



set +e
sudo rmmod gdrdrv
sudo rmmod libnvm

sudo lsmod |grep ^nvidia |awk '{print $1}' |xargs sudo rmmod
sudo lsmod |grep ^nvidia |awk '{print $1}' |xargs sudo rmmod
sudo lsmod |grep ^nvidia |awk '{print $1}' |xargs sudo rmmod


#cd /usr/src/nvidia-440.33.01
#cd /usr/src/nvidia-510.108.03
cd /usr/src/nvidia-*
#make clean
sudo IGNORE_CC_MISMATCH=1 make -j
sudo insmod nvidia.ko NVreg_OpenRmEnableUnsupportedGpus=1
sudo insmod nvidia-modeset.ko
sudo insmod nvidia-drm.ko
sudo insmod nvidia-uvm.ko

cd ${GDR_PROJECT_DIR}
sudo rm -rf /dev/gdrdrv
sudo ./insmod.sh

set -e

cd ${PROJECT_DIR}
# sudo rm -rf build
mkdir -p build; cd build
# cmake .. -DCMAKE_CUDA_ARCHITECTURES=80 
make -j

cd ${PROJECT_DIR}/build/module
make -j
sudo make load

sudo bash -c "echo -n ${ssd_pci} > /sys/bus/pci/devices/${ssd_pci}/driver/unbind"
sudo bash -c "echo -n ${ssd_pci} > /sys/bus/pci/drivers/libnvm\ helper/bind"
