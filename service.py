import tensorflow as tf

model = tf.keras.models.load_model("model")


def predict_digit_process(image):
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    predictions = model.predict(image_array)
    return predictions
