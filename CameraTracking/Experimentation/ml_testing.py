from keras.models import load_model

'Loading in the Model'
try:
     model = load_model("./Models/96_tanh_model.h5")
except:
     model = load_model("./A2E/CameraTracking/Models/96_tanh_model.h5")

print(model.summary())