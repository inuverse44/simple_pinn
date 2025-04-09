# initialize.py

from simple_PINN.settings import config

def delete_log(log_path=None):
    """
    ログファイルを削除する関数（空ファイルとして初期化）
    """
    if log_path is None:
        log_path = config.get_log_path()

    try:
        with open(log_path, "w") as f:
            f.write("")  # 空の内容で上書き（初期化）
    except FileNotFoundError:
        pass  # ファイルが存在しない場合は無視
