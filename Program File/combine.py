df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target

# Handle missing values
df = df.dropna()

# One-hot encode categorical features
df = pd.get_dummies(df)

# Define features and target variables
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)