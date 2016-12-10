# StyleTransfer
ECBM 4040 Neural Network
Deep learning Project
Group PPAP

files content:
random.py:	creates np.random 
init.py: 	creates initializers for parameter variables and contains 			distribution function
base.py:	layer and Mergelayer
normalization:	"LocalResponseNormalization2DLayer","BatchNormLayer", 			"batch_norm"
theano_extensions.py: 	used by conv.py
nonlinearities.py: nonliear functions

ECBM 4040 Neural Network &amp; 
Deep learning Project - Group PPAP
This project is referring 'A Neural Algorithm of Artistic Style'

We need to update our cuDNN
After tar the install file, cd to its path

then:
  export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
  cp lib64/* /usr/local/cuda/lib64/
  cp include/* /usr/local/cuda/include/
