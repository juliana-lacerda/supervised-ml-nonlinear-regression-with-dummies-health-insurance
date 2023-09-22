import yaml
import os

# Open the YAML configuration file
script_dir = os.path.dirname(__file__)
yaml_file = os.path.join(script_dir, "config.yaml")
with open(yaml_file, "r") as file:
    cf = yaml.safe_load(file)
