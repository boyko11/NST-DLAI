import pprint
from model_service import ModelService
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import imshow, show
import skimage

from nst_utils import *


if __name__ == '__main__':

    model_service = ModelService()

    # Start interactive session
    sess = tf.InteractiveSession()

    content_image = skimage.io.imread("images/louvre_small.jpg")
    imshow(content_image)
    show()
    content_image = reshape_and_normalize_image(content_image)

    style_image = skimage.io.imread("images/monet.jpg")
    imshow(style_image)
    show()
    style_image = reshape_and_normalize_image(style_image)

    generated_image = generate_noise_image(content_image)
    imshow(generated_image[0])
    show()

    vgg_pretrained_model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

    # Assign the content image to be the input of the VGG model.
    sess.run(vgg_pretrained_model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    out = vgg_pretrained_model['conv4_2']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = model_service.compute_content_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))

    sess.run(vgg_pretrained_model['input'].assign(style_image))

    # Compute the style cost
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]
    J_style = model_service.compute_style_cost(vgg_pretrained_model, STYLE_LAYERS, sess)
    print("J_style = " + str(J_style.eval()))

    J = model_service.total_cost(J_content, J_style, 10, 40)
    print("J = " + str(J.eval()))

    model_service.model_nn(sess, vgg_pretrained_model, J, J_content, J_style, generated_image, num_iterations=200)
