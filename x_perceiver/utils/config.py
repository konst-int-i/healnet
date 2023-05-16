"""
Config manager
"""

import yaml
import getpass
import os
import yaml
from typing import *
from box import Box

class CustomYamlLoader(yaml.FullLoader):
    """Add a custom constructor "!include" to the YAML loader.
    "!include" allows to read parameters in another YAML file as if it was
    the main one.
    Examples:
        To read the parameters listed in credentials.yml and assign them to
        credentials in logging.yml:
        ``credentials: !include credentials.yml``
        To call: config.credentials
    """

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(CustomYamlLoader, self).__init__(stream)

    def include(self, node: yaml.nodes.ScalarNode) -> Box:
        """Read yaml files as Box objects and overwrite user specific files
        Example: !include model.yml, will be overwritten by model.$USER.yml
        """

        filename: str = os.path.join(self._root, self.construct_scalar(node))
        subconfig: Box = _read(filename, loader=CustomYamlLoader)
        subconfig = _overwrite_with_user_specific_file(subconfig, filename=filename)

        return subconfig


CustomYamlLoader.add_constructor("!include", CustomYamlLoader.include)


class Config:
    def __init__(self, config_path: str):
        self._config_path = config_path

    def read(self) -> Box:
        """Reads main config file"""
        if os.path.isfile(self._config_path) and os.access(self._config_path, os.R_OK):
            config = _read(filename=self._config_path, loader=CustomYamlLoader)
            config = _overwrite_with_user_specific_file(
                config, filename=self._config_path
            )
            return config
        else:
            raise FileNotFoundError(self._config_path)

def _user_specific_file(filename: str) -> Union[None, str]:
    """Find user specific files for a filename.
    E.g. user_specific_file(config.yml) = config.$USER.yml if the file
    exists, else returns None
    """
    username = getpass.getuser().lower().replace(" ", "_")
    filepath, file_extension = os.path.splitext(filename)
    user_filename = filepath + "." + username + file_extension
    if os.path.isfile(user_filename) and os.access(user_filename, os.R_OK):
        user_filename = user_filename
    else:
        user_filename = None
    return user_filename


def _read(filename: str, loader) -> Box:
    """Read any yaml file as a Box object"""

    if os.path.isfile(filename) and os.access(filename, os.R_OK):
        with open(filename, "r") as f:
            try:
                config_dict = yaml.load(f, Loader=loader)
            except yaml.YAMLError as exc:
                print(exc)
        return Box(config_dict)
    else:
        raise FileNotFoundError(filename)


def _overwrite_with_user_specific_file(config: Box, filename: str) -> Box:
    """Overwrite the config files with user specific files """
    user_filename = _user_specific_file(filename)
    if user_filename:
        print(f"{filename} overwritten by {user_filename}")
        user_config: Box = _read(user_filename, loader=CustomYamlLoader)
        config.merge_update(user_config)

    return config



def flatten_config(dictionary, parent_key='', sep='.'):
    """
    Flatten a nested dictionary - this is required to easily update the regular config file
    with the wandb config
    Args:
        dictionary:
        parent_key:
        sep:

    Returns:
        Box: Python box object with flattened config. Elements that were previously callable via ['key']['subkey']
            are now callable via ['key.subkey']

    """
    flattened_dict = {}
    for key, value in dictionary.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(flatten_config(value, parent_key=new_key, sep=sep))
        else:
            flattened_dict[new_key] = value
    return Box(flattened_dict)