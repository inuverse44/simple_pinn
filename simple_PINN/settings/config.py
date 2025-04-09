import numpy as np

# グローバルな設定辞書（apply_configで更新）
_config = {}

# --- 設定適用用関数 ---
def apply_config(cfg):
    """
    config.yaml から読み込んだ設定を反映する
    """
    global _config
    _config = cfg.copy()

    # 初期・境界・内部点などのデータもここで生成
    _config['t_initial'] = np.zeros(cfg['N_INITIAL'])
    _config['x_initial'] = np.linspace(-1, 1, cfg['N_INITIAL'])
    _config['u_initial'] = np.sin(np.pi * _config['x_initial'])
    _config['v_initial'] = np.zeros_like(_config['x_initial'])

    _config['t_boundary'] = np.random.rand(cfg['N_BOUNDARY'])
    _config['x_boundary'] = np.random.choice([-1, 1], cfg['N_BOUNDARY'])
    _config['u_boundary'] = np.zeros(cfg['N_BOUNDARY'])

    _config['t_region'] = np.random.rand(cfg['N_REGION'])
    _config['x_region'] = np.random.uniform(-1, 1, cfg['N_REGION'])

    _config['OUTPUT_DIR'] = "output/"
    _config['TARGET_DIR'] = _config['OUTPUT_DIR'] + \
        f"init={cfg['N_INITIAL']}_boun={cfg['N_BOUNDARY']}_regi={cfg['N_REGION']}_maxep={cfg['MAX_EPOCHS_FOR_MODEL']}_lr={cfg['LEARNING_RATE']}_w={cfg['PI_WEIGHT']}_v={cfg['VELOCITY']}/"
    _config['LOG_PATH'] = _config['TARGET_DIR'] + "log.txt"


# --- Getter 関数群 ---
def get(key):
    return _config[key]

def get_target_dir():
    return _config['TARGET_DIR']

def get_log_path():
    return _config['LOG_PATH']
