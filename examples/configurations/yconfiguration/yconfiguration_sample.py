from persefone.utils.configurations import YConfiguration
from pprint import pprint

# Load nested YConfiguration
configuration = YConfiguration('main_cfg.yml')

print("\n", "*" * 10, "Raw Configuration", "*" * 10, "\n")
pprint(configuration.to_dict())

print("\n", "*" * 10, "Raw Configuration", "*" * 10, "\n")

# Replace Configuration placeholders
configuration.replace_map({
    '$PLACEHOLDER': 666.6,
    '$PLACEHOLDER2': ['one', 2, 'thre33'],
})

pprint(configuration.to_dict())
