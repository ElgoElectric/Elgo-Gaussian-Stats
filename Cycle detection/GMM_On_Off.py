import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from itertools import product


input_file_path = '/Users/visshal/Elgo-RKA/REFIT_ELGO/CLEAN_House1_extracted.csv'
df = pd.read_csv(input_file_path)
X = df[['Fridge']].values


covariance_types = ['spherical', 'diag', 'tied', 'full']


best_gmm = None
lowest_bic = np.inf
for covariance_type in covariance_types:
    gmm = GaussianMixture(
        n_components=2, covariance_type=covariance_type, random_state=0)
    gmm.fit(X)
    bic = gmm.bic(X)

    if bic < lowest_bic:
        lowest_bic = bic
        best_gmm = gmm


print(
    f"Best Model: {best_gmm.n_components} components, {best_gmm.covariance_type} covariance type")


df['Power Cycle'] = best_gmm.predict(X)


probs = best_gmm.predict_proba(X)
df['Power Cycle'] = ['ON' if prob[1] > prob[0] else 'OFF' for prob in probs]


output_file_path = '/Users/visshal/Elgo-RKA/REFIT_ELGO/CLASSIFICATION/CLEAN_House1_GMM_Tuned.csv'
df.to_csv(output_file_path, index=False)
