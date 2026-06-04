import torch
import torch.distributed as dist
import json
import os
import sys
import contextlib
import time
from typing import Optional, Dict, Any

RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)

from tools.config.recstore_config_path import find_recstore_config_path

def get_env_config() -> Dict[str, Any]:
    """Retrieves environment configuration including device count, storage type, etc."""
    config = {}
    
    # Device Info
    if torch.cuda.is_available():
        config["device_count"] = torch.cuda.device_count()
        config["device_name"] = torch.cuda.get_device_name(0)
    else:
        config["device_count"] = 0
        config["device_name"] = "CPU"
        
    # Distributed Info
    if dist.is_initialized():
        config["world_size"] = dist.get_world_size()
        config["rank"] = dist.get_rank()
    else:
        config["world_size"] = 1
        config["rank"] = 0

    # Storage Backend (RecStore specific)
    recstore_config_path = find_recstore_config_path(os.getcwd())
            
    if recstore_config_path:
        try:
            with open(recstore_config_path, 'r') as f:
                rc = json.load(f)
                cache_ps = rc.get("cache_ps", {})
                config["storage_backend_type"] = "RecStore_PS"
                config["storage_value_type"] = (
                    cache_ps.get("base_kv_config", {})
                    .get("value", {})
                    .get("type", "Unknown")
                )
        except Exception as e:
            config["storage_backend_error"] = str(e)
    else:
        config["storage_backend_type"] = "Unknown/TorchRec_Default"

    return config

def print_env_config(mode: str):
    """Prints the environment configuration in a formatted block."""
    try:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
            
        config = get_env_config()
        print("\n==========================================")
        print(f"=== {mode} Training Configuration ===")
        print("==========================================")
        print(f"Device Count:         {config['device_count']}")
        print(f"Device Name:          {config['device_name']}")
        print(f"World Size:           {config['world_size']}")
        
        if mode == "RecStore":
             print(f"Storage Backend:      {config.get('storage_backend_type', 'N/A')}")
             print(f"Value Storage Type:   {config.get('storage_value_type', 'N/A')} (DRAM=RAM, SSD=Disk)")
        else:
             print(f"Storage Backend:      Native TorchRec (HBM/UVM)")
             
        print("==========================================\n")
    except Exception as e:
        print(f"Warning: Failed to print config: {e}")

class CudaTimer:
    """Helper for timing GPU operations correctly."""
    def __init__(self, enable: bool = True):
        self.enable = enable and torch.cuda.is_available()
        if self.enable:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self.t_start = 0.0

    def __enter__(self):
        if self.enable:
            self.start_event.record()
        else:
            self.t_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            self.end_event.record()
        # Note: Must explicitly call elapsed_time() after synchronize() later
        # But for simple inline usage we might want immediate result?
        # Standard pattern: record end, but don't block. 

    def elapsed_ms(self) -> float:
        """Returns elapsed time in milliseconds. Blocks CPU if using CUDA events."""
        if self.enable:
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        else:
            return (time.time() - self.t_start) * 1000.0
