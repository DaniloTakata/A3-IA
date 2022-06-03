import pathlib

import tf
from matplotlib import pyplot as plt

from microfone import gravador
from redeNeuralTensor import model, commands, preprocess_dataset

DATASET_PATH = ''
data_dir = pathlib.Path(DATASET_PATH)

gravador()

audio_entrada = data_dir/''
audio_ds = preprocess_dataset([str(audio_entrada)])

for spectrogram, label in audio_ds.batch(1):
    prediction = model(spectrogram)
    plt.bar(commands, tf.nn.softmax(prediction[0]))
    plt.title(f'Predictions for "{commands[label[0]]}"')
    plt.show()
