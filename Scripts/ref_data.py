from librairies import *


"""
Ce script python est exécute en premier: 
      - récupere les images format JPEG pour prétraitement
      - permet de générer le 1er modele 
      - génère la PCA

"""



def load_images(anomalie, good):
  """
  Fonction qui telecharge les images depuis leurs fichiers 
  """
  images = []
  labels = []
  # Load and label 'anomalie' images
  for filename in os.listdir(good):
      if filename.endswith(('.jpg', '.png', '.jpeg')):
          img = cv2.imread(os.path.join(good, filename))
          if img is not None:
              images.append(img)
              labels.append(0)

  for filename in os.listdir(anomalie):
      if filename.endswith(('.jpg', '.png', '.jpeg')):
          img = cv2.imread(os.path.join(anomalie, filename))
          if img is not None:
              images.append(img)
              labels.append(1) 

  return images, labels


def flatten_images(images):
  """
  Prétraite l'image 
  """
  return [img.reshape(-1) for img in images]



def process_data(X_train, X_test):
  """
  Applique la PCA, Renvoie un pickel PCA dans le dossier Artifact. 
  """
  from sklearn.decomposition import PCA
  pca = PCA(n_components=36)
  X_train_pca = pca.fit_transform(X_train)
  X_test_pca = pca.transform(X_test)

  with open('Artifacts/pca.pkl','wb') as file:
    pickle.dump(pca,file)

  return X_train_pca, X_test_pca



def splitData(data, labels, method='stratify'):
  """
  Divise le dataset en mode startifié
  """
  from sklearn.model_selection import train_test_split
  return train_test_split(data, labels, test_size=0.3, random_state=99, stratify=labels)


def save_to_csv(data, p_labels, r_labels, filename):
    """
    charge le dataset dans un fichier csv    
    """
    df = pd.DataFrame(data, columns=[f'PCA_{i}' for i in range(data.shape[1])])
    df['prediction'] = p_labels  # Change 'Target' en fonction de votre structure de données
    df['target'] = r_labels

    df.to_csv(filename, index=False)



# chemins vers les fichiers Images, 
anomalie = "Data/initial_data/danomalie"
good = "Data/initial_data/good"
images, labels = load_images(anomalie, good)


#Prétraite les images
images_flattened = flatten_images(images)


#Divise le dataset
X_train, X_test, y_train, y_test = splitData(images_flattened, labels)


#Normalisation des données 
X_train = [x/255 for x in X_train ]
X_test =  [x/255 for x in X_test]


# Application de la réduction de dimension ( PCA )
X_train_NP, X_test_NP = process_data(X_train, X_test)


# Entrainement du modele 
# SVC a été choisie apres comparaison entre differents autres modeles 
clf = SVC(random_state=99)
clf.fit(X_train_NP, y_train)
pred_test = clf.predict(X_test_NP)


#Enregistrement du modele dans un Pickel
with open('Artifacts/modele.pkl', 'wb') as file:
        pickle.dump(clf, file)


#enregistre les données 
# ref_data ; correspond aux données train et test concaténées
# ref_data_test; correspond aux données test seulemnt (utilisé pour creer le rapport)
data = np.concatenate((X_train_NP, X_test_NP), axis=0)
pred_labels = np.concatenate((y_train, y_test), axis=0)
script_dir = os.path.dirname(__file__)
ref_data_path = os.path.join(script_dir, '../Data', 'ref_data.csv')
ref_test_data_path = os.path.join(script_dir, '../Data', 'ref_data_test.csv')
save_to_csv (data, pred_labels, pred_labels, ref_data_path)
save_to_csv (X_test_NP, pred_test, y_test,ref_test_data_path)














