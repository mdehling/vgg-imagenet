Training VGG on ImageNet
========================
Why?  Because for various reasons I would like to have a VGG model with the
following two optional modifications:

1. It uses average pooling instead of max pooling after the convolutional
   blocks.
2. It uses batch normalization after each convolutional layer.

Oh, and in addition I would like to use preprocessing that normalizes the
images to mean 0 and standard deviation 1 instead of what the
`keras.applications` model does.
