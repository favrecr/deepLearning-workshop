#### 2. Hello world tensorflow
What relationship you see in this numbers?

```
X: -1 0 1 2 3 4
Y: -2 1 4 7 10 13
```

```
import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

model.fit(xs, ys, epochs=10)

print(model.predict([10.0]))

```
Run the model again with 100 epochs and see if results are different.
