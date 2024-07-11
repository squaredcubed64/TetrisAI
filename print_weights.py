import tensorflow as tf

model = tf.keras.models.load_model('linear_regression.keras')

for layer in model.layers:
    weights = layer.get_weights()
    print(f"Weights for layer {layer.name}: {weights}")