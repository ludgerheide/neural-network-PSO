import json


def load_key_from_config(key):
    """
    Loads a key from config.json
    :parameter key the key to load (a string)
    :return:
    """
    json_data = open('config.json').read()
    dict = json.loads(json_data)
    return dict[key]
