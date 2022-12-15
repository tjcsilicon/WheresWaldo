import PIL.Image
import numpy as np
import tensorflow as tf
from image_slicer import slice


retrain = 0

def train(dataDir):
    batch_size = 32


    if retrain == 1:
        trainingSets, validationSets = tf.keras.utils.image_dataset_from_directory(directory=dataDir,
                                                            validation_split=0.2,
                                                            subset="both",
                                                            color_mode="rgb",
                                                            seed=123,
                                                            image_size=(64, 64),
                                                            batch_size=batch_size)

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu", input_shape=(64,64,3)),
                tf.keras.layers.MaxPooling2D((2,2)),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(100, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(2, activation="softmax")
            ]
        )
        model.summary()
        model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        checkpoint_filepath = './checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True)

        history = model.fit(x=trainingSets, epochs=30, validation_data=validationSets, batch_size=batch_size)
        loss, accuracy = model.evaluate(trainingSets, batch_size=batch_size)
        model.save(checkpoint_filepath)
        print(f"loss: {loss}\naccuracy: {accuracy}")
        print("done")
    else:
        model = tf.keras.models.load_model("./checkpoint")

    return model

def findWaldo(fileName, model): # not working
    image = PIL.Image.open(fileName)
    # image = image.convert('1')
    image = image.resize((2048, 2048))
    image.save(fileName)

    slices = slice(fileName, 1024, save=False)
    scores = []
    count = 0
    for i in slices:
        i = tf.keras.preprocessing.image.img_to_array(i.image)
        i = np.expand_dims(i, axis=0)
        prediction = model.predict(i)
        # print(prediction)
        scores.append(prediction[0][1])
        if prediction[0][1] >= 0.80:
            print("WALDO FOUND")
            slices[count].image.save("./found/10-{0}.jpg".format(count))

            # break
        count += 1

    index = scores.index(max(scores))
    print(max(scores))
    print(index)
    slices[index].image.show()


if __name__ == "__main__":
    retrain = 0 # set retrain to 1 to retrain the model.

    model = train("training/64")
    # findWaldo("testing/original-images/1.jpg", model) # Not working
    image = PIL.Image.open("tmp/waldo-test1.jpg")

    image = image.resize((64, 64))

    test_image = tf.keras.preprocessing.image.img_to_array(image)
    test_image = np.expand_dims(test_image, axis=0)
    prediction = model.predict(test_image)
    print(prediction)
