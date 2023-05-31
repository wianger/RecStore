set -x

ssd_pci="0000:5e:00.0"


umount /media/ssd1/

echo -n ${ssd_pci} > "/sys/bus/pci/devices/${ssd_pci}/driver/unbind"
