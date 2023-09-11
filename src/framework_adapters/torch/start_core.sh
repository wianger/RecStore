sudo sh -c "ulimit -c unlimited"
sudo sysctl -w kernel.core_uses_pid=1 
sudo sysctl -w kernel.core_pattern=/corefile/core.%e.%p.%s.%E
