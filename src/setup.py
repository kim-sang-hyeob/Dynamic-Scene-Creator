import os
import sys
import subprocess
import shutil

class SetupManager:
    def __init__(self, global_config, model_config):
        self.global_config = global_config
        self.model_config = model_config
        self.python_exe = sys.executable

    def run(self):
        self._check_system_dependencies()
        self._ensure_cuda_home()
        print(f"Setup Manager: Installing {self.model_config['pretty_name']}")
        
        steps = self.model_config.get('setup', [])
        for step in steps:
            cmd_template = step.get('cmd')
            check_exists = step.get('check_exists')
            cwd_template = step.get('cwd', '.')
            
            # Context for template formatting
            ctx = {
                'repo_url': self.model_config.get('repo_url'),
                'install_dir': self.model_config.get('install_dir'),
                'python': self.python_exe,
                'cuda_home': os.environ.get('CUDA_HOME', '')
            }
            
            # Check exist condition (e.g. don't clone if dir exists)
            if check_exists:
                target_path = check_exists.format(**ctx)
                if os.path.exists(target_path):
                    print(f"  [Skip] {target_path} exists.")
                    continue

            # Format command
            cmd_str = cmd_template.format(**ctx)
            cwd = cwd_template.format(**ctx)
            
            # Ensure cwd exists if we are supposed to be inside it
            if cwd != '.' and not os.path.exists(cwd):
                 # If cwd doesn't exist yet (and we aren't cloning), it's an error
                 # but usually cloning is the first step so it's fine.
                 pass

            print(f"  [Exec] {cmd_str}")
            try:
                env = os.environ.copy()
                subprocess.check_call(cmd_str, shell=True, cwd=cwd if cwd != '.' else None, env=env)
            except subprocess.CalledProcessError as e:
                print(f"  [Error] Command failed: {e}")
                sys.exit(1)
                
        print("Setup completed successfully.")

    def _check_system_dependencies(self):
        """Checks for essential system tools."""
        required_tools = ['git']
        missing = []
        for tool in required_tools:
            if not shutil.which(tool):
                missing.append(tool)
        
        if missing:
            print(f"[Error] Missing system dependencies: {', '.join(missing)}")
            print("Please install them using your package manager.")
            print(f"Example (Ubuntu/Debian): apt-get update && apt-get install -y {' '.join(missing)}")
            sys.exit(1)

    def _ensure_cuda_home(self):
        """Attempts to set CUDA_HOME if not already set, prioritizing the version matching PyTorch."""
        if os.environ.get('CUDA_HOME'):
            print(f"  [Info] Using existing CUDA_HOME: {os.environ['CUDA_HOME']}")
            return

        print("  [Info] Searching for CUDA compatible with PyTorch...")
        
        # Get Torch's preferred CUDA version first
        torch_cuda = None
        try:
            import torch
            torch_cuda = torch.version.cuda # e.g. "11.8"
            print(f"  [Info] PyTorch requires CUDA {torch_cuda}")
        except:
            pass

        nvcc_path = None

        # Strategy 1: Aggressive system-wide scan for the CORRECT nvcc version
        if torch_cuda:
            print(f"  [Info] Scanning for nvcc matching version {torch_cuda}...")
            try:
                # Find all nvcc paths
                search_cmd = "find /usr/local -name nvcc -type f 2>/dev/null"
                paths = subprocess.check_output(search_cmd, shell=True).decode().splitlines()
                
                for p in paths:
                    try:
                        # Check version of this nvcc
                        ver_out = subprocess.check_output(f"{p} --version", shell=True).decode()
                        if f"release {torch_cuda}" in ver_out:
                            nvcc_path = p
                            print(f"  [Info] Found EXACT match: {p}")
                            break
                    except:
                        continue
            except:
                pass

        # Strategy 2: Prioritize explicit version paths if scan failed
        if not nvcc_path and torch_cuda:
            p = f'/usr/local/cuda-{torch_cuda}/bin/nvcc'
            if os.path.exists(p):
                nvcc_path = p
                print(f"  [Info] Found version path: {p}")

        # Strategy 3: Falling back to PATH or other system defaults
        if not nvcc_path:
            nvcc_path = shutil.which('nvcc')
            if nvcc_path:
                print(f"  [Info] Found nvcc in PATH: {nvcc_path}")
        
        if not nvcc_path:
            fallbacks = [
                '/usr/local/cuda-11.8/bin/nvcc',
                '/usr/local/cuda-12.1/bin/nvcc',
                '/usr/local/cuda-12.2/bin/nvcc',
                '/usr/local/cuda/bin/nvcc',
                '/usr/bin/nvcc',
            ]
            for p in fallbacks:
                if os.path.exists(p):
                    nvcc_path = p
                    break

        if nvcc_path:
            # Assuming standard path structure: .../bin/nvcc
            real_path = os.path.realpath(nvcc_path)
            cuda_home = os.path.dirname(os.path.dirname(real_path))
            
            os.environ['CUDA_HOME'] = cuda_home
            # Also add to PATH so subprocesses can find it explicitly
            bin_dir = os.path.dirname(real_path)
            if bin_dir not in os.environ['PATH']:
                os.environ['PATH'] = bin_dir + os.pathsep + os.environ['PATH']

            print(f"  [Info] Final selected CUDA_HOME: {cuda_home}")
        else:
            print("  [Error] Could not detect ANY CUDA installation.")
            print("  Please find your CUDA path and run: export CUDA_HOME=/your/path/to/cuda")
            sys.exit(1)
