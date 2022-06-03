import pathlib

import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
from tkinter import *

import tf
from matplotlib import pyplot as plt

from redeNeuralTensor import model, commands, preprocess_dataset

"""
def ouvir_do_microfone():
    microfone = sr.Recognizer() # Criação de uma variável que vai receber o método de reconhecer o áudio falado.
    fs = 48000

    with sr.Microphone() as source:
        microfone.adjust_for_ambient_noise(source)

        print("Fale o comando: ")
        audio = microfone.listen(source) # Váriavel que recebero áudio inputado por completo no sistema.

    try:
        comando = microfone.recognize_google(audio, language='pt-BR').upper() #Transforma o áudio em uma strings
        print("Comando dito: " + comando)
    except sr.UnknownValueError:
        print("Comando não entendido.")

    return comando


ouvir_do_microfone()
"""


def gravador():
    fs = 48000

    duration = 1
    myrecording = sd.rec(int(duration * fs),
                         samplerate=fs, channels=2)
    sd.wait()

    return sf.write('entradas/entrada/audio/entrada.wav', myrecording, fs)


master = Tk()

Label(master, text=" Microfone de entrada: "
      ).grid(row=0, sticky=W, rowspan=5)

b = Button(master, text="Gravar", command=gravador)
b.grid(row=0, column=2, columnspan=2, rowspan=2,
       padx=5, pady=5)

mainloop()

Data_Set = "entradas/entrada"
data_dir = pathlib.Path(Data_Set)

arquivo_de_entrada = data_dir/"audio/entrada.wav"

entrada = preprocess_dataset([str(arquivo_de_entrada)])

for spectrogram, label in entrada.batch(1):
    prediction = model(spectrogram)
    plt.bar(commands, tf.nn.softmax(prediction[0]))
    plt.title(f'Predictions for "{commands[label[0]]}"')
    plt.show()
