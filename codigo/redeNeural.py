import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializador da rede
rede = Sequential()

# Adiciona a primeira camada de Convolução
rede.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Aplicação do pooling (agrupamento) na primeira saida
rede.add(MaxPooling2D(pool_size=(2, 2)))

# Adicionando a segunda camada
rede.add(Conv2D(32, (3, 3), activation='relu'))

# Aplicação do pooling (agrupamento) na segunda saida
rede.add(MaxPooling2D(pool_size=(2, 2)))

# Transforma os dados em vetores
rede.add(Flatten)

"""
Nessa parte acontecerá a conexão de todas as camadas, usando uma função de ativação retificadora (relu), logo em seguida
a sigmóide para ter as probabilidades de cada áudio ter a palavra chave.
"""
rede.add(Dense(unit=128, activation='relu'))
rede.add(Dense(unit=1, activation='sigmoid'))

"""
Logo após isso é preciso compilar a rede neural criada, utilização do otimizador "Adam", e a função loss com entropia 
binária cruzada que funciona como a sigmóide.
"""
rede.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Criação da Rede Neural pronta, agora é preciso ensina-la com exemplos.


