import pandas as pd
from joblib import load
from train import create_no_meal_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv('test.csv', header=None)

dataset = create_no_meal_matrix(data)

std_no_df = StandardScaler().fit_transform(dataset)

pca = PCA(n_components=6)
pca.fit(std_no_df)

pca_no_df = pd.DataFrame(pca.fit_transform(std_no_df))

with open('Classifier.pickle', 'rb') as pre_trained:
    pickle_file = load(pre_trained)
    predict = pickle_file.predict(pca_no_df)
    pre_trained.close()

pd.DataFrame(predict).to_csv('Result.csv', index=False, header=False)