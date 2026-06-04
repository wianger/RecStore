#!/usr/bin/env python3

import os
import sys
import argparse
import socket
import subprocess
import time
import json
import tempfile
import shutil
from contextlib import contextmanager

# Add path for ps_server_runner imports
RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)

TEST_SCRIPTS_PATH = os.path.join(RECSTORE_PATH, 'src/test/scripts')
if not os.path.exists(TEST_SCRIPTS_PATH):
    TEST_SCRIPTS_PATH = os.path.join(RECSTORE_PATH, 'test/scripts')

if TEST_SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, TEST_SCRIPTS_PATH)

try:
    from ps_server_runner import ps_server_context
    from ps_server_helpers import find_ps_server_binary
    from tools.config.recstore_config_path import resolve_recstore_config_path
except ImportError:
    print("Error: Could not import ps_server_runner. Make sure the script is in the correct location.")
    sys.exit(1)

def get_local_ip():
    """Detect the local IP address visible to the network."""
    try:
        # Connect to a public DNS server to determine the most appropriate local IP
        # This doesn't actually establish a connection
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to hostname if internet is not available
        try:
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except:
            return "127.0.0.1"

def create_temp_config(port=15000):
    """
    Create a temporary RecStore config file based on the root recstore_config.json,
    modified to bind to 0.0.0.0 and run a single shard for testing.
    """
    base_config_path = str(resolve_recstore_config_path())
    
    print(f"[Runner] Loading base config from: {base_config_path}")
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # Modify for single shard, local server bind
    # cache_ps
    if "cache_ps" not in config: config["cache_ps"] = {}
    config["cache_ps"]["num_shards"] = 1
    config["cache_ps"]["servers"] = [{
        "host": "0.0.0.0",
        "port": port,
        "shard": 0
    }]
    
    # Ensure base_kv_config uses a temp path to avoid conflicts
    if "base_kv_config" not in config["cache_ps"]: 
        config["cache_ps"]["base_kv_config"] = {
            "path": "/tmp/recstore_data_dist_test",
            "capacity": 1000000,
            "value_size": 512, 
            "value_type": "DRAM",
            "index_type": "DRAM",
            "value_memory_management": "PersistLoopShmMalloc"
        }
    else:
        # Override just the path
        config["cache_ps"]["base_kv_config"]["path"] = "/tmp/recstore_data_dist_test"

    # distributed_client
    if "distributed_client" not in config: config["distributed_client"] = {}
    config["distributed_client"]["num_shards"] = 1
    config["distributed_client"]["servers"] = [{
        "host": "0.0.0.0",
        "port": port,
        "shard": 0
    }]

    # client (simple client config if used)
    if "client" not in config: config["client"] = {}
    config["client"]["host"] = "0.0.0.0"
    config["client"]["port"] = port
    config["client"]["shard"] = 0
    
    fd, path = tempfile.mkstemp(suffix=".json", prefix="recstore_dist_config_")
    with os.fdopen(fd, 'w') as f:
        json.dump(config, f, indent=4)
    return path

def main():
    parser = argparse.ArgumentParser(description="Distributed EBC Precision Test Runner")
    parser.add_argument("--remote-host", required=True, help="Remote host in format user@ip or just ip")
    parser.add_argument("--remote-python", default="python3", help="Path to python on remote machine")
    parser.add_argument("--remote-test-path", help="Path to test_ebc_precision.py on remote machine. Defaults to same path as local.")
    parser.add_argument("--server-path", default=None, help="Path to ps_server binary. If not set, tries to find it.")
    parser.add_argument("--local-port", type=int, default=15000, help="Port for local PS Server to listen on")
    parser.add_argument("--num-embeddings", type=int, default=1000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="Force test to run on CPU")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Resolve local path for reference (and default remote path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming code layout is identical
    default_test_path = os.path.join(current_dir, 'test_ebc_precision.py')
    remote_test_path = args.remote_test_path if args.remote_test_path else default_test_path
    
    local_ip = get_local_ip()
    print(f"\n[Runner] Local IP detected as: {local_ip}")
    print(f"[Runner] Remote Host: {args.remote_host}")
    
    # Find server binary
    server_path = args.server_path
    if not server_path:
        server_path = find_ps_server_binary()
        if not server_path:
            print("Error: Could not find ps_server binary. Please build the project or specify --server-path.", file=sys.stderr)
            sys.exit(1)
    
    config_path = create_temp_config(port=args.local_port)
    print(f"[Runner] Created temporary server config at {config_path}")
    print(f"[Runner] Starting PS Server (Single Shard) on port {args.local_port}...")
    
    try:
        # Start PS Server
        # We enforce num_shards=1 because our temp config is single shard
        with ps_server_context(
            server_path=server_path, 
            config_path=config_path, 
            num_shards=1, 
            verbose=True
        ) as runner:
            print(f"[Runner] PS Server is ready and listening on {args.local_port}")
            
            # Construct Remote Command
            # Note: We pass local_ip as the ps-host for the remote client
            remote_cmd_parts = [
                args.remote_python,
                remote_test_path,
                f"--ps-host {local_ip}",
                f"--ps-port {args.local_port}",
                f"--num-embeddings {args.num_embeddings}",
                f"--embedding-dim {args.embedding_dim}",
                f"--batch-size {args.batch_size}",
                f"--seed {args.seed}"
            ]
            
            if args.cpu:
                remote_cmd_parts.append("--cpu")
                
            remote_cmd_str = " ".join(remote_cmd_parts)
            
            ssh_cmd = ["ssh", args.remote_host, remote_cmd_str]
            
            print(f"\n[Runner] Executing remote command via SSH:")
            print(f"Command: {' '.join(ssh_cmd)}")
            print("="*70)
            
            # Execute SSH
            start_time = time.time()
            try:
                subprocess.check_call(ssh_cmd)
                duration = time.time() - start_time
                print("\n" + "="*70)
                print(f"✅ Remote precision test completed successfully in {duration:.2f}s")
            except subprocess.CalledProcessError as e:
                print("\n" + "="*70)
                print(f"❌ Remote precision test failed with exit code {e.returncode}")
                sys.exit(e.returncode)
            
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.remove(config_path)
            print(f"[Runner] Removed temporary config file")

if __name__ == "__main__":
    main()
