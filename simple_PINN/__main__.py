# __main__.py
import time
from simple_PINN.simple_PINN import main_PINN
from simple_PINN.settings.cofig_loader import load_configs
from simple_PINN.settings.config import apply_config

if __name__ == '__main__':
    start_time = time.time()
    configs = load_configs("config.yaml")
    for config in configs:
        print(f"\n===== Running config: {config['name']} =====")
        apply_config(config)
        main_PINN()
    end_time = time.time()
    print(f"Execution time: (hours:minutes:seconds) {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
