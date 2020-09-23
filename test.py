import tensorflow as tf

(xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
xTrain, xTest = xTrain / 255.0, xTest / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(256, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs=5)

print("loss: %f\naccuracy: %f" % tuple(model.evaluate(xTest, yTest)))