# initialize file

from simple_PINN.settings.config import ( 
    LOG_PATH
)

def delete_log(log_path=LOG_PATH):
    """
    ログファイルを削除する関数
    """
    try:
        with open(log_path, "w") as f:
            f.write("")  # 空の内容で上書き
    except FileNotFoundError:
        pass  # ファイルが存在しない場合は無視