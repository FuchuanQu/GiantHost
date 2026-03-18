import logging
import os
import sys

def setup_logger(output_dir, log_filename="experiment.log", level=logging.DEBUG):
    """
    配置全局 Logger。
    
    Args:
        output_dir (str): 日志文件保存的目录 (通常是本次实验的 output 文件夹)。
        log_filename (str): 日志文件名。
        level (int): 日志级别 (logging.INFO, logging.DEBUG 等)。
    
    Returns:
        logger: 配置好的 logger 对象。
    """
    # 1. 确保日志目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, log_filename)

    # 2. 获取 root logger
    # 使用 root logger 可以捕获所有模块的日志，包括第三方库（如果需要的话）
    logger = logging.getLogger()
    logger.setLevel(level)

    # 3. 防止重复添加 Handler (如果在一个进程中多次调用 setup)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. 定义日志格式
    # 格式：[时间] [级别] [文件名:行号]: 信息
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 5. Handler 1: 输出到文件
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    # 格式：[时间] [级别] [文件名:行号] 信息
    file_fmt = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
    file_handler.setFormatter(file_fmt)

    # 6. Handler 2: 输出到控制台 (Console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # 格式：直接显示信息，或者加个简单的头
    console_fmt = logging.Formatter('%(message)s') 
    console_handler.setFormatter(console_fmt)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logger initialized. Saving logs to: {log_path}")
    
    return logger

def get_logger(name=None):
    """
    在其他模块中获取 logger 的快捷方式。
    虽然直接 import logging 也可以，但封装一层方便未来扩展。
    """
    return logging.getLogger(name)