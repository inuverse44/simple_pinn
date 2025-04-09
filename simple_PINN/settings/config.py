def apply_config(config):
    global N_INITIAL, N_BOUNDARY, N_REGION
    global MAX_EPOCHS_FOR_MODEL, LEARNING_RATE, PI_WEIGHT, VELOCITY
    global t_initial, x_initial, u_initial, v_initial
    global t_boundary, x_boundary, u_boundary
    global t_region, x_region

    import numpy as np

    N_INITIAL = config["N_INITIAL"]
    N_BOUNDARY = config["N_BOUNDARY"]
    N_REGION = config["N_REGION"]
    MAX_EPOCHS_FOR_MODEL = config["MAX_EPOCHS_FOR_MODEL"]
    LEARNING_RATE = config["LEARNING_RATE"]
    PI_WEIGHT = config["PI_WEIGHT"]
    VELOCITY = config["VELOCITY"]

    # 再初期化
    t_initial = np.zeros(N_INITIAL)
    x_initial = np.linspace(-1, 1, N_INITIAL)
    u_initial = np.sin(np.pi * x_initial)
    v_initial = np.zeros_like(x_initial)

    t_boundary = np.random.rand(N_BOUNDARY)
    x_boundary = np.random.choice([-1, 1], N_BOUNDARY)
    u_boundary = np.zeros(N_BOUNDARY)

    t_region = np.random.rand(N_REGION)
    x_region = np.random.uniform(-1, 1, N_REGION)

def get_target_dir():
    return f"output/init={N_INITIAL}_boun={N_BOUNDARY}_regi={N_REGION}_maxep={MAX_EPOCHS_FOR_MODEL}_lr={LEARNING_RATE}_w={PI_WEIGHT}_v={VELOCITY}/"

def get_log_path():
    return get_target_dir() + "log.txt"