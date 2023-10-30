model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)