"""
!Copyright: This Notebook is made by me the great armaan xD and i am writing this without any reason
!so Yeah Use it however you want i just wanted to add this "!Copyright" Thingy x'D
This python file basically is where i will learn about basics of tensorflow i mean this this whole folder will be where
i will learn about AI Development and stuffs 
"""
# Import Statements
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Tensors!
"""
Basically Tensors are just n-dimensional arrays which are used because they are really great in
Different Mathematical fields such as matrix multiplication and stuffs
"""

my_tensor = tf.constant(10) # This is a scalar tensor with a value of 10 in it
# To use or do any kind of thing in a tensor we have to do it inside a session

print(my_tensor) # Output: <tensorObject (...)>

with tf.Session() as sess:
    result = sess.run(my_tensor)
    print(result) # Output: 10


# Placeholders!
"""
A Placeholder is basically a tensor object which doesnt have a value at first but before computing you can 
add the value to that placeholder for example Your data in a Linear Regression Can be considered as a Placeholder
x = tf.placeholder(tf.float32) # Here We have to provide a datatype for the placeholder to work
"""
# Now for a linear regression for Example (y = m * x + b)
x = tf.placeholder(tf.float32)
m, b = 0.3, 0.7 # We start with some randome values of m and b
y = (m*x) + b  # Linear Formulae

# Even For us to use the tensor "Y" we have to run it inside a session
with tf.Session() as sess:
    result = sess.run(y, feed_dict={x: 10})
    print("Linear Result: {}".format(result)) # Output: "Linear Reult: 3.7000000000047683716"

# Just Like That We Can Actually Do Almose All The Basic Things
# Variables
"""
Now variables are used to denote hyper parameters such as m and b in a case of Linear Regression
And to use any variables we have to initialize them globally in a tensorflow graph which i will talk about later
"""

x = tf.placeholder(tf.float32)
m, b = tf.Variable(0.3), tf.Variable(0.7) # We start with some randome values of m and b
y = (m*x) + b  # Linear Formulae
# Even For us to use the tensor "Y" we have to run it inside a session
with tf.Session() as sess:
    # We have to initialize all the variables in a session that is why we are using sess.run()
    sess.run(tf.global_variables_initializer()) # This thing initializes all the variables with the ini value provided
    result = sess.run(y, feed_dict={x: 10})
    print("Linear Result: {}".format(result)) # Output: "Linear Reult: 3.7000000000047683716"

# Graphs
"""
A Graph is where tensorflow objects their operations such as addition multiplication matmul lives
for the above linear regression code the graph is like
m, x, b ----> z = m*x ---->  z + b
and when we run things inside a session it checks the graph and do the operations we run in a session
we dont have to make graphs it is set by default but we actually can make graphs but no one do that!
"""
# This just creates a linear ditribuition of data with some random noise
#-----------------------------------------------------------------------
error = np.random.randn(1, 100) * 1
X_data = np.linspace(-10, 10, 100) + error
y_data = np.linspace(-10, 10, 100)
y_data = y_data
X, Y = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
m, b = tf.Variable(0.710), tf.Variable(0.317)
#------------------------------------------------------------------------

def calculate_linear_result(learning_rate=0.0001, epochs=10):
    """
    @desc: This function trains a linear regression model
    @params: learning_rate: float; The step value to take in the gradient descent process, epochs: Int
    """
    z = (m * X) + b  # Calculate The Z Value 
    cost = tf.reduce_sum(tf.pow(Y - z, 2)) / 2 * 100 # Calculate The Cost
    optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Use GradientDescent Optimizer
    with tf.Session() as sess:
        varInit = tf.global_variables_initializer()
        sess.run(varInit)
        for i in range(epochs):
            for x_, y_ in zip(X_data[0], y_data):
                sess.run(optim, feed_dict={X: x_, Y: y_})

def show():
    """
    @desc: This Function Is Used To Plot Data
    """
    with tf.Session() as sess:
        plt.plot(X_data[0], y_data, 'x')
        sess.run(tf.global_variables_initializer())
        plt.plot(X_data[0], np.array(sess.run(tf.add(tf.multiply(m, X_data[0]), b))))
        plt.show()

calculate_linear_result()
show()
"""
So Basically In Tensorflow (v1.3.4) you have to run every thing inside a session
*placeholders are just placeholder for the data which you dont know about and will know after some time
*variables are mostly use for hyper parameters and stuffs because they can be changed
*constant are just constant tensors
*Every operation such as addition multiplication matrrix multiplcation actually takes place via something called as graph
***NOTE***
In Tensorflow 2 we dont have to run sessions infact there is no concept such as sessions in tensorflow 2 Ugh google why you did this
x'D But Yeah You Dont have to run everything inside sessions you can just directly add tensors without using sess.run()
WoW! C'X
And That almost concludes the basics of Tensorflow!
"""