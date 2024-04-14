import re

# Define the pattern according to the given logic
IS_PATTERN = re.compile(r'.*[^A-Za-z0-9_.-].*')

# Example keys to check against the pattern
keys = [
    "simpleKey",   # Expected: Does not match (contains only allowed characters)
    "complex:key", # Expected: Matches (contains a colon, which is not allowed)
    "anotherKey",  # Expected: Does not match (contains only allowed characters)
    "key-with-special@char"  # Expected: Matches (contains a special character '@')
]

# Check each key against the pattern
matches = {key: bool(IS_PATTERN.match(key)) for key in keys}

print(matches)
# With r prefix, backslashes are treated as literal characters
path = "C:\\Users\\Username\\Documents\\file.txt"
print(path)  # Outputs: C:\Users\Username\Documents\file.txt

pattern=re.compile("abc")
string="abwcd"
print(pattern.match(string))  # Output: <re.Match object; span=(0, 3), match='abc'>

import jax
import jax.numpy as jnp

out = {'a': {"123":(1,2,5)}, 'b': jnp.array([4, 5, 6])}
flat, treedef = jax.tree_util.tree_flatten(out)
flat[0]=100
print(treedef.unflatten(flat))
print(flat)

tree_map = jax.tree_util.tree_map
out2=1,2,3


def print_fn(x1, x2, x3):
    print(x1, x2, x3)

print_fn(tree_map(lambda x: x+1, out2))