import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42
DATASET_DIR = os.path.join("data", "lung")
MODEL_OUT = os.path.join("models", "lung_best_model.keras")


def build_model():
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomTranslation(0.06, 0.06),
            layers.RandomZoom(0.15, 0.15),
            layers.RandomContrast(0.15),
        ],
        name="augment",
    )

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)
    x = preprocess_input(x)

    backbone = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = False

    y = layers.GlobalAveragePooling2D()(backbone.output)
    y = layers.Dropout(0.45)(y)
    y = layers.Dense(128, activation="swish")(y)
    y = layers.Dropout(0.25)(y)
    outputs = layers.Dense(1, activation="sigmoid")(y)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model, backbone


def make_datasets():
    if not os.path.isdir(DATASET_DIR):
        raise ValueError(
            "Dataset not found. Expected folder structure: "
            "data/lung/{normal,abnormal} containing images."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
    )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1500).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds


def estimate_class_weights(dataset):
    neg = 0
    pos = 0
    for _, labels in dataset:
        y = tf.reshape(labels, [-1])
        pos += int(tf.reduce_sum(tf.cast(y > 0.5, tf.int32)).numpy())
        neg += int(tf.reduce_sum(tf.cast(y <= 0.5, tf.int32)).numpy())
    total = max(1, pos + neg)
    w0 = total / (2.0 * max(1, neg))
    w1 = total / (2.0 * max(1, pos))
    return {0: float(w0), 1: float(w1)}


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.35),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def train():
    os.makedirs("models", exist_ok=True)
    train_ds, val_ds = make_datasets()
    class_weights = estimate_class_weights(train_ds)
    model, backbone = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUT, monitor="val_pr_auc", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=2, min_lr=1e-6
        ),
    ]

    compile_model(model, 1e-3)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=18,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    backbone.trainable = True
    for layer in backbone.layers[:-35]:
        layer.trainable = False

    compile_model(model, 2e-5)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=14,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    print(f"Best lung model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    train()
