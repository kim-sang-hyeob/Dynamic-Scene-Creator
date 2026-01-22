import os
import sys
import subprocess
import shutil

class Runner:
    def __init__(self, global_config, model_config):
        self.global_config = global_config
        self.model_config = model_config
        self.python_exe = sys.executable
        self.output_root = global_config['paths']['output_root']

    def train(self, scene_path, extra_args=""):
        scene_name = os.path.basename(os.path.normpath(scene_path))
        # Separate output folder for each model type to avoid overwrites
        model_name = self.model_config['name']
        output_path = os.path.join(self.output_root, model_name, scene_name)
        
        print(f"Runner: Training {self.model_config['pretty_name']}")
        print(f"Scene : {scene_path}")
        print(f"Output: {output_path}")

        train_cfg = self.model_config.get('train', {})
        cmd_template = train_cfg.get('command')
        cwd_template = train_cfg.get('cwd', '.')
        
        if not cmd_template:
            print("[Error] No train command defined for this model.")
            return

        ctx = {
            'install_dir': self.model_config.get('install_dir'),
            'python': self.python_exe,
            'scene_path': os.path.abspath(scene_path),
            'output_path': os.path.abspath(output_path),
            'extra_args': extra_args
        }
        
        cmd_str = cmd_template.format(**ctx)
        cwd = cwd_template.format(**ctx)
        
        print(f"  [Exec] {cmd_str}")

        # Defensive check: ensure external directory exists
        if cwd != '.' and not os.path.exists(cwd):
            print(f"[Error] Model directory not found: {cwd}")
            print(f"Please run: python manage.py setup --model {model_name}")
            sys.exit(1)

        try:
            os.makedirs(output_path, exist_ok=True)
            # Set environment variable to resolve MKL/libgomp conflict
            env = os.environ.copy()
            env["MKL_THREADING_LAYER"] = "GNU"
            subprocess.check_call(cmd_str, shell=True, cwd=cwd if cwd != '.' else None, env=env)
            self._post_process(output_path, scene_name)
        except subprocess.CalledProcessError as e:
            print(f"  [Error] Training failed: {e}")
            sys.exit(e.returncode)

    def _post_process(self, output_path, scene_name):
        """Copies the final .ply file to the server's static directory for web viewing."""
        static_dir = os.path.join("server", "static")
        os.makedirs(static_dir, exist_ok=True)
        
        # Look for .ply files in the output directory
        point_cloud_dir = os.path.join(output_path, "point_cloud")
        if os.path.exists(point_cloud_dir):
            # Find the iteration folder (usually 'iteration_7000' or similar)
            iters = [d for d in os.listdir(point_cloud_dir) if os.path.isdir(os.path.join(point_cloud_dir, d))]
            if iters:
                latest_iter = sorted(iters, key=lambda x: int(x.split('_')[-1]) if '_' in x else 0)[-1]
                ply_src = os.path.join(point_cloud_dir, latest_iter, "point_cloud.ply")
                if os.path.exists(ply_src):
                    ply_dst = os.path.join(static_dir, f"{scene_name}.ply")
                    shutil.copy2(ply_src, ply_dst)
                    print(f"[Runner] Exported result to {ply_dst}")
