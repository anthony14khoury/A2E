from sklearn.metrics import silhouette_score, silhouette_samples
from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model
from sklearn.cluster import KMeans
from parameters import Params
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
     
def gathering_data(folder, letters, label_map):
     sequences, labels = [], []
    
     for letter in letters:
          print(letter)
          
          dir_length = len(os.listdir(os.path.join(folder, letter)))
          
          for i in range(0, dir_length):

               # Grab all 30 frames and append them to window
               res = np.load(os.path.join(folder, letter, letter + str(i) + ".npy"))
               sequences.append(res)
               labels.append(label_map[letter])
               # print(os.path.join(folder, letter, letter + str(i) + ".npy"))
    
     return sequences, labels


'Loading in Parameters and Letters'
params = Params()
letters = params.LETTERS
label_map = {label:letters for letters, label in enumerate(letters)}

'Grabbing all of the Data'
folder = "./A2E/CameraTracking/DataCollection"
sequences, labels = gathering_data(folder, letters, label_map)
data = np.array(sequences)

'Loading in the Model'
try:
     model = load_model("./Models/96_tanh_model.h5")
except:
     model = load_model("./A2E/CameraTracking/Models/96_tanh_model.h5")

     
'Initializing Model'
layers = model.layers
final_layer = layers[-1]
new_model = Model(inputs=model.input, outputs=final_layer.output)

plot_model(new_model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

features = new_model.predict(data)

'Reshaping Data'
x, y, z = data.shape
data = data.reshape(x, y*z)
     
'Perform k-means clustering'
clusterer = KMeans(n_clusters=len(letters), random_state=42)
cluster_labels = clusterer.fit_predict(features)

'Overall Silhouette Score'
silhouette_avg = silhouette_score(features, cluster_labels, metric='euclidean')
print("Silhoutte Score: {}".format(silhouette_avg))

