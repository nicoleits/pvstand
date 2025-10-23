# config/logging_config.py

import logging
import sys

def setup_logger(name='soiling_analysis_logger', level=logging.WARNING):
    """
    Configura y retorna un logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar añadir múltiples handlers si el logger ya tiene
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Handler para la consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Opcional: Handler para archivo (si se necesita en el futuro)
        # file_handler = logging.FileHandler('soiling_analysis.log')
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)

    return logger

def set_log_level(level):
    """
    Cambia el nivel de logging dinámicamente.
    
    Args:
        level: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

# Crear una instancia global del logger para ser importada por otros módulos
logger = setup_logger() 