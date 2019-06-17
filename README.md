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

- Model:
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

  # Set optimizer
  optimizer = SGD(lr=1e-2, momentum=0.9, decay=1e-6)
  
  # Compile Model
  model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
  
  EPOCHS = 70
  ES_PATIENCE = 10 # EarlyStopping patience value
  RLROP_PATIENCE = 3 # ReduceLROnPlateau patience value
  RLROP_FACTOR = 0.2 # ReduceLROnPlateau factor value

  callbacks = [ModelCheckpoint(filepath="grabai_cars.Xception_FT.weights.best.hdf5", 
                               monitor="val_loss", 
                               save_best_only=True, 
                               verbose=1),
               EarlyStopping(monitor="val_acc", 
                             patience=ES_PATIENCE, 
                             min_delta=1e-3, 
                             restore_best_weights=True,
                             verbose=1),
               ReduceLROnPlateau(monitor="val_loss", 
                                 patience=RLROP_PATIENCE, 
                                 factor=RLROP_FACTOR, 
                                 verbose=1)
              ]

  # Train the model
  history = model.fit_generator(train_generator,
                      steps_per_epoch=train_steps,
                      epochs=EPOCHS, verbose=1, callbacks=callbacks,
                      validation_data=valid_generator,
                      validation_steps=valid_steps)
  ```
