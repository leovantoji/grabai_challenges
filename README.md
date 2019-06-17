## Grab AI Computer Vision Challenge
- The model appears to suffer from overfitting issue.

  |Dataset|Images|Accuracy|
  |:-:|:-:|:-:|
  |Training|6515|93.05%|
  |Validation|815|68.22%|
  |Testing|814|70.76167%|
  |Public Test|8041|72.60291%|

- My Computer:
  - OS: Ubuntu 18.04.2 LTS 64-bit
  - Memory: 15.6 GiB
  - Processor: Intel® Core™ i5-4460 CPU @ 3.20GHz × 4
  - Graphics: GeForce RTX 2070

- Model Architecture:
  ```python
  Xception_model = Xception(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
  
  # Fine-tune Xception from layer 94 onwards
  for i, layer in enumerate(Xception_model.layers):
    if i < 94: layer.trainable = False
    else: layer.trainable = True
  
  model = Sequential()
  model.add(Xception_model)
  model.add(BatchNormalization())
  model.add(GlobalAveragePooling2D())
  model.add(Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
  model.add(Dropout(rate=0.5))
  model.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
  model.add(Dropout(rate=0.5))
  model.add(Dense(N_CLASSES, activation="softmax"))
  ```
