import abc
import argparse
import hashlib
import json
import os
import yaml
from mergedeep import merge, Strategy
import re


def value_parser(value):
    SPECIAL_KEY = "SPECIAL_KEY"
    if re.match("^[-+]?[0-9]*\\.?[0-9]+(e[-+]?[0-9]+)?$", value) is None:
        return str(value)
    else:
        return yaml.safe_load(f"{SPECIAL_KEY}: {value}")[SPECIAL_KEY]


def merge_parameters(a, b):
    return merge(a, b, strategy=Strategy.REPLACE)


class Parser(abc.ABC, metaclass=abc.ABCMeta):
    def do(self, parameter_value: str):
        return self._do(parameter_value)

    @abc.abstractmethod
    def _do(self, parameter_value: str) -> dict:
        raise NotImplementedError


class ConfigFileParser(Parser):
    def __init__(self):
        self.file_parsers = {
            ".yaml": yaml.full_load,
            ".yml": yaml.full_load,
            ".json": json.load,
        }

    def _do(self, parameter_value: str) -> dict:
        path = parameter_value
        extension = os.path.splitext(path)[1]
        with open(path, "r") as f:
            return self.file_parsers[extension](f)


class StrParser(Parser):
    @staticmethod
    def key_value_to_dict(key, value):
        splits = key.split(".", maxsplit=1)
        current_key = splits[0]
        next_key = splits[1] if len(splits) > 1 else None
        if next_key is None:
            dictionary = {current_key: value}
        else:
            dictionary = {current_key: StrParser.key_value_to_dict(next_key, value)}
        return dictionary

    def _do(self, parameter_value: str) -> dict:
        splits = str(parameter_value).split("=", maxsplit=1)
        key, value = splits[0], value_parser(splits[1])
        return StrParser.key_value_to_dict(key, value)


class InlineJsonParser(Parser):
    def _do(self, parameter_value: str) -> dict:
        return json.loads(str(parameter_value))


def get_config():
    parser = argparse.ArgumentParser()

    parser_actions = {
        parser.add_argument(
            "-c",
            help="Provide config file in yaml or json.",
            action="append",
        ).dest: ConfigFileParser(),
        parser.add_argument(
            "-j",
            help="Inline json.",
            action="append",
        ).dest: InlineJsonParser(),
        parser.add_argument(
            "-p",
            help="Provide extra parameters on the form key1.key2[key3]=value.",
            action="append",
        ).dest: StrParser(),
    }
    args = vars(parser.parse_args())
    current_parameters = {}

    for key in args:
        parser_action = parser_actions[key]
        if args[key] is not None:
            for value in args[key]:
                merge_parameters(current_parameters, parser_action.do(value))

    return current_parameters


def get_config_hash():
    return get_dict_hash(get_config())


def get_dict_hash(dictionary):
    dictionary_str = json.dumps(dictionary, sort_keys=True)
    dictionary_hash = hashlib.md5(dictionary_str.encode("utf-8")).hexdigest()
    return dictionary_hash


def save_config(pre_path):
    with open(f"{pre_path}{get_config_hash()}.yaml", "w") as f:
        yaml.dump(get_config(), f)
