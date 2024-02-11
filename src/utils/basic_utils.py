"""
summary
"""

import textwrap
from os.path import normpath

import yaml
from box import Box

from src.exception import CustomException
from src.logger import logger


def read_yaml(yaml_path: str) -> Box:
    """
    This function reads a YAML file from the provided path and returns
    its content as a Box object.

    Args:
        yaml_path (str): The path to the YAML file to be read.

    Raises:
        CustomException: If there is any error while reading the file or
        loading its content, a CustomException is raised with the original
        exception as its argument.

    Returns:
        Box: The content of the YAML file, loaded into a Box object for
        easy access and manipulation.
    """
    try:
        yaml_path = normpath(yaml_path)
        with open(yaml_path, encoding="utf-8") as yf:
            content = Box(yaml.safe_load(yf))
            logger.info("yaml file: %s loaded successfully", yaml_path)
            return content
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e


def wrap_text(text: str, wrap_width: int = 110) -> str:
    """_summary_

    Args:
        text (str): _description_
        wrap_width (int, optional): _description_. Defaults to 110.

    Returns:
        str: _description_
    """
    # Split the input text into lines based on newline characters
    lines = text.split("\n")

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=wrap_width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = "\n".join(wrapped_lines)

    return wrapped_text
