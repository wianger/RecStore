#!/usr/bin/env python3

import os
import sys
import time
import signal
import subprocess
import threading
import argparse
import re
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Callable


class PSServerRunner:
    def __init__(
        self,
        server_path: str = "./build/bin/ps_server",
        config_path: Optional[str] = None,
        log_dir: str = "/tmp",
        timeout: int = 120,
        num_shards: int = 2,
        verbose: bool = False,
        startup_delay: float = 2.0
    ):
        self.server_path = Path(server_path)
        self.config_path = Path(config_path) if config_path else None
        self.log_dir = Path(log_dir)
        self.timeout = timeout
        self.num_shards = num_shards
        self.verbose = verbose
        self.startup_delay = startup_delay
        
        self.process: Optional[subprocess.Popen] = None
        self.processes: List[subprocess.Popen] = []
        self.log_file = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.output_threads: List[threading.Thread] = []
        self.ready_event = threading.Event()
        self.shard_ready = set()
        self._stopping = False
        self._suppressing_shutdown_stderr = False
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_log_file(self):
        import getpass
        try:
            user = getpass.getuser()
        except:
            user = "unknown"
            
        script_name = os.path.basename(sys.argv[0])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_filename = f"ps_server_{user}_{script_name}_{timestamp}.log"
        log_path = self.log_dir / log_filename
        
        self.log_file = open(log_path, 'w', buffering=1)
        return log_path
    
    def _log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] {message}\n"
        if self.log_file:
            self.log_file.write(log_line)
        if self.verbose:
            print(log_line.rstrip())

    def _resolve_config_path(self) -> Optional[Path]:
        if self.config_path:
            return self.config_path

        env_config = os.environ.get("RECSTORE_CONFIG")
        if env_config:
            return Path(env_config)

        default_config = Path("/app/RecStore/recstore_config.json")
        if default_config.exists():
            return default_config

        return None

    def _configured_shard_ids(self) -> List[int]:
        config_path = self._resolve_config_path()
        if not config_path or not config_path.exists():
            return list(range(self.num_shards))

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            self._log(f"⚠ Failed to read shard ids from {config_path}: {e}")
            return list(range(self.num_shards))

        servers = config.get("cache_ps", {}).get("servers", [])
        shard_ids = []
        if isinstance(servers, list):
            for server in servers:
                if isinstance(server, dict) and isinstance(server.get("shard"), int):
                    shard_ids.append(server["shard"])

        if shard_ids:
            return shard_ids

        return list(range(self.num_shards))

    def _build_command(self, shard_id: Optional[int] = None) -> List[str]:
        cmd = [str(self.server_path)]
        if self.config_path:
            cmd.extend(["--config_path", str(self.config_path)])
            self._log(f"Config: {self.config_path}")
        if shard_id is not None:
            cmd.extend(
                [
                    f"--grpc_local_shard_id={shard_id}",
                    f"--local_shard_id={shard_id}",
                ]
            )
        return cmd

    def _start_process(self, cmd: List[str], label: str) -> subprocess.Popen:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(self.server_path.parent.parent.parent)
        )

        self.processes.append(process)
        self.process = process
        self._log(f"Server process started ({label}, PID: {process.pid})")

        stdout_thread = threading.Thread(
            target=self._monitor_output,
            args=(process.stdout, "STDOUT"),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=self._monitor_output,
            args=(process.stderr, "STDERR"),
            daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()

        self.output_threads.extend([stdout_thread, stderr_thread])
        self.stdout_thread = stdout_thread
        self.stderr_thread = stderr_thread
        return process

    def _wait_for_shard_ready(
        self, shard_id: int, process: subprocess.Popen, start_time: float
    ) -> bool:
        check_interval = 0.5
        while shard_id not in self.shard_ready:
            if time.time() - start_time > self.timeout:
                self._log(f"✗ Timeout waiting for shard {shard_id} to be ready")
                self._log(f"   Detected shards: {sorted(self.shard_ready)}")
                self.stop()
                return False

            if process.poll() is not None:
                self._log(
                    f"✗ Server process for shard {shard_id} terminated "
                    f"unexpectedly (exit code: {process.returncode})"
                )
                self.stop()
                return False

            time.sleep(check_interval)

        return True

    def _should_suppress_shutdown_stderr(self, line: str) -> bool:
        if not self._stopping:
            self._suppressing_shutdown_stderr = False
            return False

        block_start_markers = (
            "*** Aborted at ",
            "*** Signal 15 (SIGTERM)",
            "folly::symbolizer::",
        )
        if any(marker in line for marker in block_start_markers):
            self._suppressing_shutdown_stderr = True
            return True

        if self._suppressing_shutdown_stderr:
            stack_frame_prefixes = (
                "    @ ",
                "\t@ ",
                "                       /",
                "/app/RecStore/",
            )
            if line.startswith(stack_frame_prefixes):
                return True
            self._suppressing_shutdown_stderr = False

        return False

    def _monitor_output(self, pipe, stream_name: str):
        try:
            for line in iter(pipe.readline, ''):
                if not line:
                    break
                line = line.rstrip()

                if stream_name == "STDERR" and self._should_suppress_shutdown_stderr(line):
                    continue

                self._log(f"[{stream_name}] {line}")
                
                if "listening on" in line:
                    try:
                        shard_ids = [
                            int(m.group(1))
                            for m in re.finditer(
                                r"(?:bRPC\s+)?Server shard\s+(\d+)\s+listening on",
                                line,
                            )
                        ]
                        if not shard_ids and "Server listening on" in line:
                            shard_ids = [0]
                        if not shard_ids:
                            continue

                        for shard_id in shard_ids:
                            self.shard_ready.add(shard_id)
                            self._log(f"✓ Detected shard {shard_id} ready")

                        if len(self.shard_ready) >= self.num_shards:
                            self._log(f"✓ All {self.num_shards} shards ready")
                            self.ready_event.set()
                    except (IndexError, ValueError) as e:
                        self._log(f"⚠ Failed to parse shard info: {e}")
        except Exception as e:
            self._log(f"✗ Monitor thread error: {e}")
        finally:
            pipe.close()
    
    def start(self) -> bool:
        if not self.server_path.exists():
            raise FileNotFoundError(f"Server binary not found: {self.server_path}")
        
        log_path = self._create_log_file()
        self._log(f"Starting PS Server: {self.server_path}")
        self._log(f"Log file: {log_path}")

        try:
            start_time = time.time()

            if self.num_shards > 1:
                shard_ids = self._configured_shard_ids()
                self._log(
                    f"Starting {len(shard_ids)} PS server shards sequentially: "
                    f"{shard_ids}"
                )

                for shard_id in shard_ids:
                    cmd = self._build_command(shard_id=shard_id)
                    self._start_process(cmd, f"shard {shard_id}")
                    self._log(
                        f"Waiting for shard {shard_id} to be ready "
                        f"(timeout: {self.timeout}s)..."
                    )
                    if not self._wait_for_shard_ready(
                        shard_id, self.process, start_time
                    ):
                        return False
            else:
                cmd = self._build_command()
                self._start_process(cmd, "single")

                time.sleep(self.startup_delay)

                self._log(
                    f"Waiting for {self.num_shards} shards to be ready "
                    f"(timeout: {self.timeout}s)..."
                )

                check_interval = 0.5
                while not self.ready_event.is_set():
                    if time.time() - start_time > self.timeout:
                        self._log(f"✗ Timeout waiting for server to be ready")
                        self._log(
                            f"   Detected {len(self.shard_ready)}/"
                            f"{self.num_shards} shards: {sorted(self.shard_ready)}"
                        )
                        self.stop()
                        return False

                    if self.process.poll() is not None:
                        self._log(
                            f"✗ Server process terminated unexpectedly "
                            f"(exit code: {self.process.returncode})"
                        )
                        self.stop()
                        return False

                    time.sleep(check_interval)

            time.sleep(0.5)
            
            self._log("✓ Server is ready")
            return True
                
        except Exception as e:
            self._log(f"✗ Failed to start server: {e}")
            self.stop()
            raise
    
    def stop(self):
        if self.process is None and not self.processes:
            return
        
        self._log("Stopping PS Server...")
        self._stopping = True
        self._suppressing_shutdown_stderr = False
        
        try:
            for process in self.processes:
                if process.poll() is None:
                    process.terminate()

            for process in self.processes:
                if process.poll() is not None:
                    continue
                try:
                    process.wait(timeout=5)
                    self._log(
                        f"✓ Server process {process.pid} terminated gracefully"
                    )
                except subprocess.TimeoutExpired:
                    self._log(
                        f"⚠ Server process {process.pid} didn't terminate, "
                        "killing..."
                    )
                    process.kill()
                    process.wait()
                    self._log(f"✓ Server process {process.pid} killed")
        except Exception as e:
            self._log(f"✗ Error stopping server: {e}")
        
        for thread in self.output_threads:
            thread.join(timeout=2)
        
        if self.log_file:
            self.log_file.close()
            self._log = lambda msg: None
        
        self.process = None
        self.processes = []
        self.output_threads = []
        self.stdout_thread = None
        self.stderr_thread = None
        self._stopping = False
        self._suppressing_shutdown_stderr = False
    
    def is_running(self) -> bool:
        return bool(self.processes) and all(
            process.poll() is None for process in self.processes
        )
    
    @contextmanager
    def run(self):
        success = self.start()
        if not success:
            raise RuntimeError("Failed to start PS Server")
        
        try:
            yield self
        finally:
            self.stop()


@contextmanager
def ps_server_context(
    server_path: str = "./build/bin/ps_server",
    config_path: Optional[str] = None,
    log_dir: str = "/tmp",
    timeout: int = 120,
    num_shards: int = 2,
    verbose: bool = False,
    startup_delay: float = 2.0
):
    runner = PSServerRunner(
        server_path=server_path,
        config_path=config_path,
        log_dir=log_dir,
        timeout=timeout,
        num_shards=num_shards,
        verbose=verbose,
        startup_delay=startup_delay
    )
    
    with runner.run():
        yield runner


def run_with_server(
    test_func: Callable,
    server_path: str = "./build/bin/ps_server",
    config_path: Optional[str] = None,
    log_dir: str = "/tmp",
    timeout: int = 30,
    num_shards: int = 2,
    verbose: bool = False
):
    with ps_server_context(
        server_path=server_path,
        config_path=config_path,
        log_dir=log_dir,
        timeout=timeout,
        num_shards=num_shards,
        verbose=verbose
    ):
        return test_func()


def main():
    parser = argparse.ArgumentParser(
        description="PS Server Test Runner - Manages ps_server lifecycle for tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server and keep it running
  python3 ps_server_runner.py --keep-alive
  
  # Run a test command with server
  python3 ps_server_runner.py --command "python3 test.py"
  
  # Use custom config and log directory
  python3 ps_server_runner.py --config recstore_config.json --log-dir ./test_logs --keep-alive
  
  # In Python code:
  from ps_server_runner import ps_server_context
  
  with ps_server_context():
      # Run your tests here
      run_tests()
        """
    )
    
    parser.add_argument(
        "--server-path",
        default="./build/bin/ps_server",
        help="Path to ps_server binary (default: ./build/bin/ps_server)"
    )
    parser.add_argument(
        "--config",
        help="Path to recstore config file"
    )
    parser.add_argument(
        "--log-dir",
        default="/tmp",
        help="Directory for server logs (default: /tmp)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds waiting for server ready (default: 30)"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=2,
        help="Expected number of server shards (default: 2)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print server output to console"
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep server running (for interactive testing)"
    )
    parser.add_argument(
        "--command",
        help="Command to run after server starts"
    )
    
    args = parser.parse_args()
    
    runner = PSServerRunner(
        server_path=args.server_path,
        config_path=args.config,
        log_dir=args.log_dir,
        timeout=args.timeout,
        num_shards=args.num_shards,
        verbose=args.verbose or args.keep_alive
    )
    
    try:
        if not runner.start():
            print("✗ Failed to start server", file=sys.stderr)
            return 1
        
        if args.command:
            print(f"\n{'='*70}")
            print(f"Running command: {args.command}")
            print(f"{'='*70}\n")
            
            result = subprocess.run(args.command, shell=True)
            exit_code = result.returncode
            
            print(f"\n{'='*70}")
            print(f"Command exited with code: {exit_code}")
            print(f"{'='*70}\n")
            
            return exit_code
        
        elif args.keep_alive:
            print(f"\n{'='*70}")
            print("Server is running. Press Ctrl+C to stop.")
            print(f"{'='*70}\n")
            
            try:
                while runner.is_running():
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\nReceived interrupt, stopping server...")
        
        return 0
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
    
    finally:
        runner.stop()


if __name__ == "__main__":
    sys.exit(main())
