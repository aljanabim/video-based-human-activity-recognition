from tensorflow.keras.models import load_model
import os


##USGE
# Replace tf.keras.models.load_model(path) with loading.model_load(path)
# In this way we keep the functionalities such as mode.evaluate, model.predict and model.fit

def model_load(path):
    if path[-1] =="/":
        weights = path + "weights"
    else:
        weights = path + "/weights"

    model = load_model(path)
    if not os.path.exists(weights):
        print("[INFO] Saving the weights. This will happen only the first time.")
        model.save_weights(weights)

    model.load_weights(weights)
    return model