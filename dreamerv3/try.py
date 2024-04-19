# import re

# # Define the pattern according to the given logic
# IS_PATTERN = re.compile(r'.*[^A-Za-z0-9_.-].*')

# # Example keys to check against the pattern
# keys = [
#     "simpleKey",   # Expected: Does not match (contains only allowed characters)
#     "complex:key", # Expected: Matches (contains a colon, which is not allowed)
#     "anotherKey",  # Expected: Does not match (contains only allowed characters)
#     "key-with-special@char"  # Expected: Matches (contains a special character '@')
# ]

# # Check each key against the pattern
# matches = {key: bool(IS_PATTERN.match(key)) for key in keys}

# print(matches)
# # With r prefix, backslashes are treated as literal characters
# path = "C:\\Users\\Username\\Documents\\file.txt"
# print(path)  # Outputs: C:\Users\Username\Documents\file.txt

# pattern=re.compile("abc")
# string="abwcd"
# print(pattern.match(string))  # Output: <re.Match object; span=(0, 3), match='abc'>

# import jax
# import jax.numpy as jnp

# out = {'a': {"123":(1,2,5)}, 'b': jnp.array([4, 5, 6])}
# flat, treedef = jax.tree_util.tree_flatten(out)
# flat[0]=100
# print(treedef.unflatten(flat))
# print(flat)

# tree_map = jax.tree_util.tree_map
# out2=1,2,3


# def print_fn(x1, x2, x3):
#     print(x1, x2, x3)

# print_fn(tree_map(lambda x: x+1, out2))

# import jax
# import jax.numpy as np

# x = np.arange(4).reshape(1, -1)
# x2= np.arange(4).reshape(1, -1)+1000
# # x=np.concat([x,x2],axis=0)
# y = jax.pmap(lambda x: jax.lax.pmean(x, 'i'), axis_name='i')(x)
# print(y)

# per = np.percentile
# x=np.concatenate([x,x2],axis=0)
# res=per(x,50.0)
# print(res)
# assert res==2.0, "Error"
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# # Assume a tensor of shape [10, 20, 3], where 3 is originally the event dimension
# # Each element in 20 is considered an independent event, but you want to treat them as a single event
# dist = tfd.Normal(loc=tf.zeros([10, 20, 3]), scale=1)  # Shape [10, 20, 3]
# independent_dist = tfd.Independent(dist, reinterpreted_batch_ndims=1)  

# # The Independent wraps these [20] independent normals into a single multi-dimensional Normal distribution
# # Sample from the distribution
# sample = independent_dist.sample()

# # Compute log probability of the sample
# log_prob = independent_dist.log_prob(sample)

# print("Sample shape:", sample.shape)  # This will output (10, 20,3)
# print("Log probability shape:", log_prob.shape)  # This will output (10,20)

# poisson_2_by_3 = tfd.Poisson(
#     rate=[[1., 10., 100.,], [2., 20., 200.]],
#     name='Two-by-Three Poissons')
# res=poisson_2_by_3.log_prob(tf.constant([1., 2.])[..., tf.newaxis, tf.newaxis])
# print(res)
# aa=[-2.9957323 , -0.10536051, -0.16251892, -1.609438  , -0.2876821 ]
# print(sum(aa))

import re
print(re.match("image", "image_layer1"))  # Match found
print(re.match("image", "my_image_layer1"))  # No match
print(re.match("$^", ""))
# print(".*")

# shapes={
#     'a':1,
#     'b':2
# }       
# print(shapes.items())        
# some_key, some_shape = list(shapes.items())[0]
# print((1,)+(2,3,4))

# needs = f'{{{", ".join(shapes.keys())}}}'
# needss=f'abc{{{shapes.keys()}}}'

# print(needs,needss)
import jax.numpy as jnp
# # x=jnp.array([[1,2],[5,6],[9,10],[13,14]])

# # H,W=x.shape
# # x = x.reshape(( H // 2, W // 2, 4))
# # print(x)
import jax.lax
# x=jnp.array([[1,2,3],[5,6,7],[9,10,11]],dtype=jnp.float32)
# kernel_size=(2,2)
# strides=(2,2)
# out=jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, kernel_size, strides, 'same')
# rhs=jnp.array([[1,2],[1,1]],dtype=jnp.float32)[...,jnp.newaxis,jnp.newaxis]
# x=x[jnp.newaxis,...,jnp.newaxis]
# out_conv=jax.lax.conv_general_dilated(x, rhs, window_strides=(2, 2), padding='SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
# # print(out)
# print(out_conv.shape)

# low=0.0
# high=1.0
logits=jnp.array([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]],dtype=jnp.float32)

# ss=tfd.OneHotCategorical(logits=logits)

# print("logits shape",logits.shape)
# ss=jnp.split(logits, 2, -1)
# print("splitted",ss)
# new_logits=logits[...,None]
# newnew_logits=new_logits[:,None]
# print("new_logits shape",newnew_logits.shape)
# split_indices=jnp.array([1,2],dtype=jnp.int32)
# yy=jnp.split(logits,split_indices,axis=-1)
# print(yy)

x=jnp.array([0.1,0.2,0.3],dtype=jnp.float32)
y=x[..., None]
print(jnp.sum(logits))
# # print(jnp.concatenate([x,y],axis=-1))
# bins = jnp.linspace(low, high, logits.shape[-1])
# print("bins",bins)
# below = (bins <= x[..., None]).astype(jnp.int32).sum(-1) - 1
# print(below)

# import tensorflow_probability as tfp
# import tensorflow as tf

# # Enable eager execution for immediate result outputs (not needed if using TF2.x).
# tf.compat.v1.enable_eager_execution()

# tfd = tfp.distributions

# # Define probabilities for each category/class.
# # These should sum to 1 across the last dimension.
# probs = [0.1, 0.6, 0.3]  # Example probabilities for 3 categories.
# probs=jnp.array(probs,dtype=jnp.float32)
# probs=probs[None,...]
# # Create a OneHotCategorical distribution with specified probabilities.
# dist = tfd.OneHotCategorical(probs=probs)

# # Sample from the distribution.
# sample = dist.sample()

# # Print the output sample and its shape.
# print("Sample:")
# print(sample.numpy())
# print("Sample Shape:", sample.shape)

# # Print the probabilities and their shape.
# print("Probabilities:")
# print(dist.probs_parameter().numpy())
# print("Probabilities Shape:", dist.probs_parameter().shape)

# aaa=()
# print(aaa is None)
# x=-0.5
# silu=x*jax.nn.sigmoid(x)
# print(x,silu)

# wd_pattern=r'/(w|kernel)$'
# pat=re.compile(wd_pattern)
# aa=pat.search("12/kernel")
# print(bool(aa))

# ss='good'
# print(ss.lstrip('g'))
# from dreamerv3.jaxutils import tree_keys

# exp_dict={'a':x,'b':x,'c':{"d":x,"w":x}}
# print(tree_keys(exp_dict))

# tree_map = jax.tree_util.tree_map # This is a function from the jax.tree_util module that applies a given function to each element in a nested structure (such as lists, tuples, dictionaries, etc.) in a recursive manner
# sg = lambda x: tree_map(jax.lax.stop_gradient, x)
# res=tree_map(lambda k: bool(pat.search(k)), tree_keys(exp_dict))
# print(res)

# import optax
# transform1 = optax.scale_by_adam()
# transform2 = optax.scale(-0.1)
# chained_transform = optax.chain(transform1, transform2)
# params = {'a': 1.0}
# state = chained_transform.init(params)
# updates = {'a': -0.5}
# updates, new_state = chained_transform.update(updates, state, params)
# print(updates)
# print(new_state)

# IS_PATTERN = re.compile(r'.*[^A-Za-z0-9_.-].*')
# IS_PATTERN2 = re.compile(r'.ab_nabc$')  # \ will be basically ignored
# print(bool(IS_PATTERN2.match(".ab_nabc")))  
# k="cba"
# print(f'abc{k}\{k}')

nested_dict = {'a': {'b\\': {'c\\': 1}}}
print(str(nested_dict))
lines = str(nested_dict).split(':')[2:]
print('\n'.join('--' + re.sub(r'[:,\\]', '', x) for x in lines))

#\\\\\\\\Attention////////// re.sub(r'[:,\\]' correct, re.sub('[:,\\]' raise error: Reason---the only difference is \\ will become \, and if without r, then the resulting \ will be further as escape for the last ], causing error