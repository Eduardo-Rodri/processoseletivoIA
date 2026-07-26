import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

import os
import tensorflow as tf


def main():
    model = tf.keras.models.load_model("model.h5")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    h5_size = os.path.getsize("model.h5") / 1024
    tflite_size = os.path.getsize("model.tflite") / 1024

    print("Conversão concluída.")
    print(f"Tamanho model.h5:     {h5_size:.1f} KB")
    print(f"Tamanho model.tflite: {tflite_size:.1f} KB")
    print(f"Redução de tamanho:   {(1 - tflite_size / h5_size) * 100:.1f}%")


if __name__ == "__main__":
    main()