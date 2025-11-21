import yaml

epoch_config = """
    reject:
        grad: 4000e-13
        mag: 4e-12
"""

# Parse YAML configuration
config = yaml.safe_load(epoch_config)

print(config)
