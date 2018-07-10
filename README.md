# DenseNet121 Trained with 1 or 2 GPU
A deep learning performance script for Keras using a TensorFlow backend.  It loads pretrained weights from the current directory and then trains for 2 epochs on synthetic data.  The result should be a quick script (runs in a few seconds) with which you can evaluate multi-gpu training 

# Required Download
As mentioned above, this network assumes a file (densenet121_weights_tf_dim_ordering_tf_kernels.h5) contianing the pre-trained weights for DenseNet121 is in the same directory as the python script.  The file is about 32MB and so is too big to uploaded here.  However it is freely available from via:
[direct download](https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels.h5)
-or-
via wget:
  sudo wget https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels.h5

# Addtional Resource
I would be remiss if I did not cite numerous other pre-trained weights available here:
[Francois Chollet's various pretrained weights](https://github.com/fchollet/deep-learning-models/releases)
