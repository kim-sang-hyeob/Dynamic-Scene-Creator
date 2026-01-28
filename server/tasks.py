import os
import sys
import subprocess
from celery import Celery

# Initialize Celery
# Note: Use redis as broker (e.g. redis://localhost:6379/0)
broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app = Celery("3dgs_worker", broker=broker_url)

# Add project root to path so we can import src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.sfm_utils import auto_process_video
from src.runner import Runner
from src.model_registry import ModelRegistry

@celery_app.task(name="process_3dgs_job")
def process_3dgs_job(job_id, video_path, model_name, scene_name):
    """
    Worker task: 
    1. Extract Frames & Run COLMAP
    2. Run Training
    3. Update Job Status
    """
    print(f"[Worker] Starting Job {job_id}...")
    
    # In a real app, you'd use a shared DB to update status. 
    # Here we log or use a global dict if in same process (but worker is separate).
    
    try:
        # 1. Setup paths
        data_root = "data"
        scene_dir = os.path.join(data_root, scene_name)
        
        # 2. Run SfM Pipeline
        print(f"  -> Extracting cameras for {scene_name}")
        success = auto_process_video(video_path, scene_dir)
        
        if not success:
            raise Exception("SfM pipeline failed.")
            
        # 3. Start Training
        print(f"  -> Training with model {model_name}")
        registry = ModelRegistry()
        model_config = registry.get_model(model_name)
        
        # Load global config (mimicking manage.py)
        global_config = {
            'paths': {'output_root': 'output', 'data_root': 'data'}
        }
        
        runner = Runner(global_config, model_config)
        runner.train(scene_dir)
        
        print(f"[Worker] Job {job_id} completed successfully.")
        return True
        
    except Exception as e:
        print(f"[Worker] Job {job_id} failed: {e}")
        return False
