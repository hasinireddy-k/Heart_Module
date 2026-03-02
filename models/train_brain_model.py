import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, callbacks, models

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_HEAD = 8
EPOCHS_FINETUNE = 10
SEED = 42

DATA_ROOT = os.path.join("brain_module", "data")
YES_DIR = os.path.join(DATA_ROOT, "yes")
NO_DIR = os.path.join(DATA_ROOT, "no")
OUTPUT_MODEL = os.path.join("models", "brain_model.h5")


def build_dataset():
    image_paths = []
    labels = []

    for name in sorted(os.listdir(YES_DIR)):
        path = os.path.join(YES_DIR, name)
        if os.path.isfile(path):
            image_paths.append(path)
            labels.append(1)

    for name in sorted(os.listdir(NO_DIR)):
        path = os.path.join(NO_DIR, name)
        if os.path.isfile(path):
            image_paths.append(path)
            labels.append(0)

    if not image_paths:
        raise ValueError("No training images found in brain_module/data.")

    image_paths = np.array(image_paths)
    labels = np.array(labels, dtype=np.int32)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    return train_paths, val_paths, train_labels, val_labels


def parse_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.float32)


def make_loader(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        augmenter = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.08),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.1),
            ]
        )
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model, base


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    train_paths, val_paths, train_labels, val_labels = build_dataset()
    train_ds = make_loader(train_paths, train_labels, training=True)
    val_ds = make_loader(val_paths, val_labels, training=False)

    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_labels,
    )
    class_weights = {0: float(class_weights_values[0]), 1: float(class_weights_values[1])}

    model, base = build_model()

    cbs = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.4, patience=2, min_lr=1e-6),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=cbs,
    )

    base.trainable = True
    for layer in base.layers[:-25]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weights,
        callbacks=cbs,
    )

    os.makedirs("models", exist_ok=True)
    model.save(OUTPUT_MODEL)
    print(f"Saved improved model to: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
