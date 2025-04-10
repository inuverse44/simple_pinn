from simple_PINN.settings import config

def delete_log(log_path=None):
    """
    Initialize or delete the log file by overwriting it with empty content.

    If no path is specified, the default log path is retrieved from the current config.
    If the file does not exist, the operation is safely skipped.

    Parameters:
        log_path (str, optional): Path to the log file. Defaults to config.get_log_path().

    Returns:
        None
    """
    if log_path is None:
        log_path = config.get_log_path()

    try:
        with open(log_path, "w") as f:
            f.write("")  # Overwrite with empty content
    except FileNotFoundError:
        pass  # Ignore if the file does not exist
