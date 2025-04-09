# entry point for the package
if __name__ == '__main__':
    from simple_PINN.simple_PINN import main_PINN
    from simple_PINN.settings.config_loader import load_configs
    from simple_PINN.settings import config as config_module
    configs = load_configs("config.yaml")

    for config in configs:
        print(f"\n===== Running config: {config['name']} =====")
        config_module.apply_config(config)  # ← 設定反映
        main_PINN()  # ← 引数なしでOK
        print(f"===== Finished config: {config['name']} =====\n")

