from librairies import *


def save_to_csv(data, p_labels, r_labels, filename):
    df = pd.DataFrame(data, columns=[f'PCA_{i}' for i in range(data.shape[1])])
    df['prediction'] = p_labels  
    df['target'] = r_labels

    df.to_csv(filename, index=False)

  

#importation des données produite et initial 
prod_data_path = '/Data/prod_data.csv'
ref_data_path = '/Data/ref_data.csv'
prod_data = pd.read_csv(prod_data_path)
ref_data = pd.read_csv(ref_data_path)
concatenated_data = pd.concat([ref_data, prod_data]).drop(columns=['prediction'])

#division des données
x_train, x_test, y_train, y_test = train_test_split(concatenated_data.drop(columns=['target'],axis=1), concatenated_data['target'], test_size=0.3, random_state=99)


#Entrainement du modele
clf = SVC(random_state=99)
clf.fit(x_train, y_train)
pred_test = clf.predict(x_test)


#save the modele
with open('/Artifacts/modele.pkl', 'wb') as file:
        pickle.dump(clf, file)


# sauvegare pred_data ( j'ecrase l'ancien )
data = np.concatenate((x_train, x_test), axis=0)
pred_labels = np.concatenate((y_train, y_test), axis=0)



#sauvegarde pred_data_test
script_dir = os.path.dirname(__file__)
ref_data_path = os.path.join(script_dir, '../Data', 'ref_data.csv')
ref_test_data_path = os.path.join(script_dir, '../Data', 'ref_data_test.csv')
save_to_csv (data, pred_labels, pred_labels, ref_data_path)
save_to_csv (x_test, pred_test, y_test,ref_test_data_path)



print(' MODEL UPDATED')
