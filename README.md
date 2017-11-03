The (mini)VGGnet architecture

In both ShallowNet and LeNet we have applied a series of CONV => RELU => POOL layers. However, in VGGNet, we stack multiple CONV => RELU layers prior to applying a single POOL layer.
Doing this allows the network to learn more rich features from the CONV layers prior to downsampling the spatial input size via the POOL operation.
Overall, MiniVGGNet consists of two sets of CONV => RELU => CONV => RELU => POOL layers, followed by a set of FC => RELU => FC => SOFTMAX layers. The first two CONV layers
will learn 32 filters, each of size 3× 3. The second two CONV layers will learn 64 filters, again, each of size 3× 3. Our POOL layers will perform max pooling over a 2× 2 window with a 1× 1 stride.
We’ll also be inserting batch normalization layers after the activations along with dropout layers(DO) after the POOL and FC layers.
The network architecture itself is detailed in Table 15.1, where the initial input image size isassumed to be 32× 32× 3 as we’ll be training MiniVGGNet on CIFAR-10 later in this chapter and then comparing performance to ShallowNet).

Again, notice how the batch normalization and dropout layers are included in the network
architecture based on my “Rules of Thumb”  Applying batch normalization will
help reduce the effects of overfitting and increase our classification accuracy on CIFAR-10.

To run this program :

python minivggnet_dataset.py --output file/output.png
