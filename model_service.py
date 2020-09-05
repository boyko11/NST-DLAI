import tensorflow as tf
from skimage import io

class ModelService:

    def __init__(self):
        pass

    def compute_content_cost(self, a_C, a_G):
        """
        Computes the content cost

        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

        Returns:
        J_content -- scalar that you compute using equation 1 above.
        """

        # Retrieve dimensions from a_G (≈1 line)
        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshape a_C and a_G (≈2 lines)
        a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
        a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])

        # compute the cost with tensorflow (≈1 line)
        J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)

        return J_content


    def gram_matrix(self, A):
        """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)

        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """

        GA = tf.matmul(A, tf.transpose(A))

        return GA

    def compute_layer_style_cost(self, a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

        Returns:
        J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
        """

        # Retrieve dimensions from a_G (≈1 line)
        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
        a_S = tf.transpose(tf.reshape(a_S, shape=[n_H * n_W, n_C]))
        a_G = tf.transpose(tf.reshape(a_G, shape=[n_H * n_W, n_C]))

        # Computing gram_matrices for both images S and G (≈2 lines)
        GS = self.gram_matrix(a_S)
        GG = self.gram_matrix(a_G)

        # Computing the loss (≈1 line)
        J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / ((2 * n_C * n_H * n_W) ** 2)

        return J_style_layer

    def compute_style_cost(self, model, STYLE_LAYERS, sess):
        """
        Computes the overall style cost from several chosen layers

        Arguments:
        model -- our tensorflow model
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them

        Returns:
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """

        # initialize the overall style cost
        J_style = 0

        for layer_name, coeff in STYLE_LAYERS:
            # Select the output tensor of the currently selected layer
            out = model[layer_name]

            # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
            a_S = sess.run(out)

            # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
            # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
            # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
            a_G = out

            # Compute style_cost for the current layer
            J_style_layer = self.compute_layer_style_cost(a_S, a_G)

            # Add coeff * J_style_layer of this layer to overall style cost
            J_style += coeff * J_style_layer

        return J_style

    def total_cost(self, J_content, J_style, alpha=10, beta=40):
        """
        Computes the total cost function

        Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost

        Returns:
        J -- total cost as defined by the formula above.
        """

        J = alpha * J_content + beta * J_style

        return J

    def model_nn(self, sess, vgg_pretrained_model, J, J_content, J_style, input_image, num_iterations=200):

        # define optimizer (1 line)
        optimizer = tf.train.AdamOptimizer(2.0)

        # define train_step (1 line)
        train_step = optimizer.minimize(J)

        # Initialize global variables (you need to run the session on the initializer)
        sess.run(tf.global_variables_initializer())

        # Run the noisy input image (initial generated image) through the model. Use assign().
        sess.run(vgg_pretrained_model['input'].assign(input_image))

        for i in range(num_iterations):

            # Run the session on the train_step to minimize the total cost
            sess.run(train_step)

            # Compute the generated image by running the session on the current model['input']
            generated_image = sess.run(vgg_pretrained_model['input'])

            # Print every 20 iteration.
            if i % 20 == 0:
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))

                # save current generated image in the "/output" directory
                _, rows, cols, channels = generated_image.shape
                io.imsave("output/" + str(i) + ".png", generated_image.reshape((rows, cols, channels)))

        # save last generated image
        _, rows, cols, channels = generated_image.shape
        io.imsave('output/generated_image.jpg', generated_image.reshape((rows, cols, channels)))

        return generated_image


