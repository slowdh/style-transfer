from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model


def tensor_to_image(tensor):
    np_image = np.array(tensor * 255, dtype=np.uint8)[0,...]
    return Image.fromarray(np_image)

def get_vgg_model():
    vgg = tf.keras.applications.VGG19(include_top=False)
    vgg.trainable=False
    return vgg

def get_transfer_model(vgg_model, layers):
    vgg_model.trainable = False
    transfer_model = Model([vgg_model.input], [vgg_model.get_layer(name).output for name in layers])
    return transfer_model

def get_gram_matrix(tensor):
    _, nh, nw, nc = tensor.shape
    unrolled = tf.reshape(tensor, (-1, nc))
    gram = tf.matmul(tf.transpose(unrolled), unrolled)
    return gram

def compute_style_cost(noise_output, style_output):
    _, n_h, n_w, n_c = noise_output.shape
    gram_noise = get_gram_matrix(noise_output)
    gram_style = get_gram_matrix(style_output)
    factor = 1 / (4 * (n_h * n_w * n_c) ** 2)
    cost = factor * tf.reduce_sum((gram_noise - gram_style) ** 2)
    return cost

def compute_content_cost(noise_output, content_output):
    _, n_h, n_w, n_c = noise_output.shape
    cost = 1 / (4 * n_h * n_w * n_c) * tf.reduce_sum((noise_output - content_output) ** 2)
    return cost

def precompute_outputs(model, input_tensor):
    outputs = model(input_tensor)
    return outputs

def split_style_content_outputs(outputs, num_content=1):
    style = outputs[:-1 * num_content]
    content = outputs[-1 * num_content:]
    return (style, content)

def merge_multiple_costs(costs, weights=None):
    if weights is None:
        weights = tf.constant([1 / len(costs) for _ in range(len(costs))])
    total = tf.reduce_sum(costs * weights)
    return total

def get_total_loss(noise_contents, noise_styles, precomputed_content_outputs, precomputed_style_outputs, alpha=1e4, beta=1e-2):
    content_costs = [compute_content_cost(noise, content) for noise, content in zip(noise_contents, precomputed_content_outputs)]
    style_costs = [compute_style_cost(noise, style) for noise, style in zip(noise_styles, precomputed_style_outputs)]
    total_content_loss = merge_multiple_costs(content_costs)
    total_style_loss = merge_multiple_costs(style_costs)
    total = alpha * total_content_loss + beta * total_style_loss
    return total

def generate_noise_image(content_img, ratio=0.9):
    _, c_h, c_w, c_c = content_img.shape
    noise_image = tf.convert_to_tensor(np.random.uniform(0, 1, (1, c_h, c_w, c_c)))
    image = noise_image * ratio + content_img * (1 - ratio)
    return image

def clip_0_1(image):
    clipped = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
    return clipped

def train_step(extractor, noise, precomputed_content_outputs, precomputed_style_outputs, optimizer, alpha=1e4, beta=1e-2):
    with tf.GradientTape() as tape:
        preprocessed_noise = tf.keras.applications.vgg19.preprocess_input(noise * 255.)
        noise_outputs = extractor(preprocessed_noise)
        noise_styles, noise_contents = split_style_content_outputs(noise_outputs)
        loss = get_total_loss(noise_contents, noise_styles, precomputed_content_outputs, precomputed_style_outputs, alpha, beta)

    grad = tape.gradient(loss, noise)
    optimizer.apply_gradients([(grad, noise)])
    noise.assign(clip_0_1(noise))
    return noise, loss

def preprocess_input(img):
    return tf.keras.applications.vgg19.preprocess_input(img)


# load VGG-19 model
vgg_model = get_vgg_model()

# get extractor model
layers = ['block1_conv1',
          'block2_conv1',
          'block3_conv1',
          'block4_conv1',
          'block5_conv1',
          'block5_conv2']

extractor = get_transfer_model(vgg_model, layers)

# load content and style image
img_size = (224, 224)
content_img = Image.open('tf2/datasets/sample_images/me.jpeg')
content_img = content_img.resize(img_size)
content = tf.expand_dims(tf.convert_to_tensor(np.array(content_img.resize(img_size))[..., :3], dtype=tf.float32), axis=0)
content_outputs = extractor(content)
_, precomputed_content_outputs = split_style_content_outputs(content_outputs)

style_img = Image.open('tf2/datasets/sample_images/monet_800600.jpg')
style = tf.expand_dims(tf.convert_to_tensor(np.array(style_img.resize(img_size))[..., :3], dtype=tf.float32), axis=0)
styles = extractor(style)
precomputed_style_outputs, _ = split_style_content_outputs(styles)

# generate random image
noise = tf.convert_to_tensor(generate_noise_image(content, ratio=0))
noise = tf.cast(noise, tf.float32)
noise = tf.Variable(noise)

# lets make fancy image!
optimizer = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.99, epsilon=1e-1)

print('Start training')
for i in range(100):
    noise, loss = train_step(extractor, noise, precomputed_content_outputs, precomputed_style_outputs, optimizer, alpha=1e4, beta=1e-2)
    print(i, loss)
    if i % 10 == 0:
        img = tensor_to_image(noise)
        img.show()
