housing = fetch_openml('house_prices')

print(housing.data.shape)
print(housing.target.shape)
print(housing.feature_names)