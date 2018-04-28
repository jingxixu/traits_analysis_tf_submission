'''

----------------------------------------------------------------------
-- Utility Lib for network operations

-- Qiangeng Xu
----------------------------------------------------------------------

'''

import tensorflow as tf

# qx2128


def linear(input_, output_size, name, stddev=0.02, bias_start=0.0,
           reuse=tf.AUTO_REUSE, with_w=False):
    '''
         This is a function automatically creates fully connected layer given input tensor and output dimension.
    '''
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name, reuse=reuse):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def conv2d(input_, output_dim,
           k_h=4, k_w=4, d_h=1, d_w=1, stddev=0.02,
           name="conv2d", reuse=False, padding='same'):
    '''
         This is a function automatically creates 2d convolution layer given input tensor and output dimension.
         the kernel size and stride is optional
    '''
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

        return conv


def conv3d(input_, output_dim,
           k_d=1, k_h=3, k_w=3, s_d =1, s_h=1, s_w=1, stddev=0.02,
           name="conv3d", reuse=False, padding='SAME'):
    '''
         This is a function automatically creates 3d convolution layer given input tensor and output dimension.
         the kernel size and stride is optional
    '''
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv3d(input_, w, strides=[1, s_d, s_h, s_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv

def relu(x):
    '''
         This is a function automatically creates relu layer
    '''
    return tf.nn.relu(x)

def lrelu(x, leak=0.2, name="lrelu"):
    '''
         This is a function automatically creates leaky relu layer. The negative slop is set
         to 0.2 as default following convention in DCGAN
    '''
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)



def batch_norm(inputs, name, train=True, reuse=False):
    '''
         This is a function automatically creates batchnorm layers
    '''
    return tf.contrib.layers.batch_norm(inputs=inputs,is_training=train,
                                      reuse=reuse,scope=name,scale=True)

def MaxPooling(x, shape, stride=None, padding='VALID'):
  """ This is a helper function creates MaxPooling layers on 4 dimensional input.
   input: tensor.
   shape: int or [h, w]
   stride: int or [h, w]. default to be shape.
   padding: 'valid' or 'same'. default to 'valid'
   NHWC tensor.
  """
  padding = padding.upper()
  shape = shape4d(shape)
  if stride is None:
    stride = shape
  else:
    stride = shape4d(stride)

  return tf.nn.max_pool(x, ksize=shape, strides=stride, padding=padding)


def shape4d(a):
    '''This is a helper function expand 2d shape to 4d shape with heading and trailing ones '''
    return [1] + shape2d(a) + [1]


def shape2d(a):
  """
   This is a helper function expand 1d shape to 2d shape
  """
  if type(a) == int:
      return [a, a]
  if isinstance(a, (list, tuple)):
      assert len(a) == 2
      return list(a)
