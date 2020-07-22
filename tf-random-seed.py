import tensorflow as tf

# If neigher the global seed nor the operation seed is set,
# we get different results for every call to the random op and every re-run
# of the program:

#print(tf.random.uniform([1]))  #0.3928  #re-run 0.1734
#print(tf.random.uniform([1]))  #0.5748  #re-run 0.2172


# If the global sed is set but the operation seed is not set,
# we got different results for every call to the random op,
# but the same sequence for every re-run of the program:
# Thre reason we get '0.3253' instead '0.5380' on the second call
# is because the second call uses a different operation seed.

#tf.random.set_seed(1234)
#print(tf.random.uniform([1]))  #0.5380  #re-run 0.5380
#print(tf.random.uniform([1]))  #0.3253  #re-run 0.3253


# Note that tf.function act like a re-run of a program in this case.
# When the global seed is set but operation seeds are not set,
# the sequence of random numbers are the same for each tf.function:

tf.random.set_seed(1234)

@tf.function
def f():
    a = tf.random.uniform([1])
    b = tf.random.uniform([1])
    return a, b

@tf.function
def g():
    a = tf.random.uniform([1])
    b = tf.random.uniform([1])
    return a, b

print(f())  #0.1304, 0.1689
print(g())  #0.1304, 0.1689


