import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def load_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0


    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)


def build_model():

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),


        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),


        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),


        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),


        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), padding="same"),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),  # Regularização antes da saída
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.1,  # Split explícito treino/validação
        callbacks=[early_stop],
        verbose=2,
    )

    val_accuracy = max(history.history["val_accuracy"])
    print(f"\nAcurácia de validação final (melhor): {val_accuracy:.4f}")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Acurácia no conjunto de teste: {test_acc:.4f}")

    model.save("model.h5")
    print("Modelo salvo em model.h5")


if __name__ == "__main__":
    main()