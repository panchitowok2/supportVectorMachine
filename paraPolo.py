import pandas as pd
from sklearn.decomposition import PCA

# Read the dataset
data = pd.read_csv('dataset.csv', sep=",")

# Replacing categorical values with numerical values
#data['Classes'] = data['Classes'].replace({'fire': 1, 'not fire': -1})

# Separate features from the target variable
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a new DataFrame with PCA-transformed features and the target variable
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Classes'] = y

# Save the DataFrame to a new CSV file
df_pca.to_csv('dataset_pca.csv', index=False)