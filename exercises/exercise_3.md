## Taller de construcción de Redes Generativas Antagónicas (RGAs)
### 
Machine Learning, Aprendizaje Automático, TensorFlow, TensorBoard, Keras, Redes de Neuronas, GAN Networks. 

## Ejercicio 3 - Construcción de una red de neuronas para la generación de números manuscritos

El objetivo de este ejercicio es construir una Red Generativa Antagónica (RGA) a partir de ejemplos aleatorios o imágenes existentes. Para ello vamos a basar este taller en una serie de talleres previos creados por Google. El objetivo de este taller es crear un sistema compuesto por lo siguientes componentes:

<img src="../img/estructura_del_ejercicio.png" alt="Estructura del ejercicio" width="800"/>

* Model: Es una clase que contendrá todos los elementos de la RGA a construir. En este caso nuestro modelo estará a su vez formado por dos modelos de red de neuronas de tipo convolucional, ya que nuestro objetivo es generar imágenes que representan a caracteres numéricos. 
* Train: Es un proceso de utilizará el modelo para entrenar los diferentes componentes que se encuentran definidos en el modelo. 

**Paso 1: Instalando paquetes en Notebooks**

Los notebooks son entidades independientes que permiten la utilización de cualquier tipo de páquete python y para ellos nos ofrece la posibilidad de instalar paquete mediante la utilización de la sistema de instalación de paquetes pip. En el caso de que estés utilizando un Jupyter Notebook es necesario realizar la instalación de ciertos paquetes mediante los siguiente comandos:

```
!sudo apt update
!pip install tensorflow matplotlib numpy Pillow==2.2.2 keras
```

Como podemos observar, es necesario incluir el caracter __!__ antes del comando de instalación. A continuación hay que seleccionar el fragmento y pulsar la tecla play para ejecutar el código contenido en el fragmento. Si ejecutamos estos comandos en el entorno colaborate observaremos la siguiente salida:

```
Requirement already satisfied: tensorflow in /usr/local/lib/python3.6/dist-packages (2.3.0)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (3.2.2)
Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (1.18.5)
Requirement already satisfied: Pillow in /usr/local/lib/python3.6/dist-packages (2.2.2)
Requirement already satisfied: keras in /usr/local/lib/python3.6/dist-packages (2.4.3)
Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.15.0)
Requirement already satisfied: scipy==1.4.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.4.1)
Requirement already satisfied: h5py<2.11.0,>=2.10.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (2.10.0)
Requirement already satisfied: tensorboard<3,>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (2.3.0)
Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.35.1)
Requirement already satisfied: absl-py>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.10.0)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (3.3.0)
Requirement already satisfied: gast==0.3.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.3.3)
Requirement already satisfied: keras-preprocessing<1.2,>=1.1.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.1.2)
Requirement already satisfied: tensorflow-estimator<2.4.0,>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (2.3.0)
Requirement already satisfied: grpcio>=1.8.6 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.32.0)
Requirement already satisfied: google-pasta>=0.1.8 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.2.0)
Requirement already satisfied: wrapt>=1.11.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.12.1)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.1.0)
Requirement already satisfied: astunparse==1.6.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.6.3)
Requirement already satisfied: protobuf>=3.9.2 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (3.12.4)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (2.4.7)
Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (2.8.1)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (0.10.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (1.2.0)
Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from keras) (3.13)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow) (1.7.0)
Requirement already satisfied: google-auth<2,>=1.6.3 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow) (1.17.2)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow) (0.4.1)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow) (3.2.2)
Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow) (50.3.0)
Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow) (1.0.1)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow) (2.23.0)
Requirement already satisfied: rsa<5,>=3.1.4; python_version >= "3" in /usr/local/lib/python3.6/dist-packages (from google-auth<2,>=1.6.3->tensorboard<3,>=2.3.0->tensorflow) (4.6)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.6/dist-packages (from google-auth<2,>=1.6.3->tensorboard<3,>=2.3.0->tensorflow) (0.2.8)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from google-auth<2,>=1.6.3->tensorboard<3,>=2.3.0->tensorflow) (4.1.1)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<3,>=2.3.0->tensorflow) (1.3.0)
Requirement already satisfied: importlib-metadata; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from markdown>=2.6.8->tensorboard<3,>=2.3.0->tensorflow) (1.7.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<3,>=2.3.0->tensorflow) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<3,>=2.3.0->tensorflow) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<3,>=2.3.0->tensorflow) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<3,>=2.3.0->tensorflow) (1.24.3)
Requirement already satisfied: pyasn1>=0.1.3 in /usr/local/lib/python3.6/dist-packages (from rsa<5,>=3.1.4; python_version >= "3"->google-auth<2,>=1.6.3->tensorboard<3,>=2.3.0->tensorflow) (0.4.8)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<3,>=2.3.0->tensorflow) (3.1.0)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata; python_version < "3.8"->markdown>=2.6.8->tensorboard<3,>=2.3.0->tensorflow) (3.1.0)
```

En este caso no se ha realizado la instalación de ningún paquete debido a que todos los paquetes necesarios ya estaban instalados en el servidor Jupyter. 

**Paso 2: Iniciando TensorBoard**

A continuación vamos a incluir un comando que permite cargar la extensión de TensorFlow Board dentro de los cuadernos de tipo Jupyter, de forma que se despligue de manera embebida en el entorno.

```
%load_ext tensorboard
```
**Paso 2: Importando Paquetes**

Una vez que se ha realizado la instalación de los diferentes paquetes python, es necesario importar aquellas clases y métodos necesarios para la realización del ejercicio.

```
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import datetime

from PIL import Image
from IPython import display
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Dropout, Flatten
from keras import optimizers

```

Para el desarrollo de los diferentes ejercicios vamos a necesitar un conjunto de liberías que servirán para lo siguiente:

* tensorflow: Nos ofrece funciones para la construcción de los modelos de Machine Learning. 
* numpy: Nos ofrece funciones para la manipulación de arrays y conjunto de datos. 
* matplotlib: Nos ofrece funciones para la visualización de datos. 
* tensorflow: Nos ofrece funciones para la construacción de procesos de entrenamiento. 
* os: Nos ofrece funciones para la manipulación de recursos del sistema operativo. 
* math: Nos ofrece funciones para la realización de operaciones matemáticas complejas (no elementales).
* time: Nos ofrece funciones para la obtención de information referente al tiempo, para crear contadores o archivos de log.
* Keras.model: Nos permite utilizar diferentes tipo de modelos, en este caso vamos a utilizar el modelo secuencial.
* Keras.layers: Nos permite utilizar diferentes tipo de capas para incluir en una red de neuronas.
* optimizers from keras: Nos permite utilizar diferentes tipos de algoritmos de optimización, en nuestro caso utilizaremos el optimizador de Adams.

**Paso 4: Definición de variable globales**

Para la construcción de nuestra red RGA es necesario definir una serie de variables globales que utilizaremos a lo largo del notebook con el objetivo de simplificar los cambios que queramos realizar en el proceso de construcción de las redes. Las variables globales a definir son las siguientes:

```
BUFFER_SIZE = 60000
BATCH_SIZE = 256
WIDTH = 28
HEIGHT = 28
```

**Paso 5 Carga de los datos**

Una vez que hayamos definido todos los elementos básicos necesarios para el taller vamos ha descargar nuestro datos. En este caso vamos a utilizar los datos de ejemplo almacenados en keras para la identificación de número manuscritos. Para ellos deberemos crear la siguiente función 

```
def generate_data_set(height, weight, normalization_value):
  (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
  train_images = train_images.reshape(train_images.shape[0], height, weight, 1).astype('float32')
  train_images = (train_images - normalization_value) / normalization_value
  return train_images, train_labels
```
Esta función crear un conjunto de datos y modificar el tamaño de las mismas en base al tamaño que queramos utilizar (reshape). Además permite normalizar el valor de los datos en base al escalado realizado en el proceso de reshape mediante la variable de normalización. Finalmente esta función nos devolverá dos conjunto de datos: (1) el conjunto de imágenes de entrenamiento; y (2) el conjunto de valores de test que nos indican que número se encuentra escrito en las imágenes. 

**Paso 6: Creación del modelo generador (Generator)**

A continuación podemos comenzar a crear le primera de las redes de neuronas necesarias para nuestro sistema. Esta red se corresponde con el generador que será la red que nos proveerá de ejemplos generados a partir de información aleatorios con el objetivo de conseguir generar ejemplos que se adapten a nuestras necesidades. Es decir, esta red es la que deber imitar el proceso de escritura manuscrita. La red está formada por las diferentes capas que se presentan en la imagen:

<img src="../img/red_generadora.png" alt="Estructura de la red generadora" width="800"/>

Esta red estará formada por 11 capas de las cuales 3 de ellas se corresponde con una función de activación y para su construcción crearemos una función que llamaremos en nuestra clase Model. A continuación se describen las diferentes capas:

* Capa Full-Connected (Dense)
* Capa Reshape (Reshape)
* Cada de transposición (Conv2DTranspose)
* Capa de activación ReLu (LeakyReLU)
* Capa de Normalización (BatchNormalization)

Para la definición del proceso de creación de la red de neuronas de generación (Generator) debemos crear una función denominada __build_generator_model__ que tendrán cuatro parámetros de entrada:

* name (string): Este parámetro se corresponde con el identificador del modelo. 
* height (int): Este parámetro se corresponde con la altura de la imagen. 
* width (int): Este parámetro se corresponde con la anchura de la imagen.
* dims (int): Se corresponde con el número de canales de la imagen. En este taller vamos a trabajar con imágenes en blanco y negro (monocromo) por lo que sólo tendremos un canal. 

```
def build_generator_model(name, height, width, dims=1):
    
    model = Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(height*width, dims)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model
```

**Paso 7: Creación del modelo generador (Generator)**

A continuación podemos comenzar a crear le primera de las redes de neuronas necesarias para nuestro sistema. Esta red se corresponde con el generador que será la red que nos proveerá de ejemplos generados a partir de información aleatorios con el objetivo de conseguir generar ejemplos que se adapten a nuestras necesidades. Es decir, esta red es la que deber imitar el proceso de escritura manuscrita. La red está formada por las diferentes capas que se presentan en la imagen:

<img src="../img/red_generadora.png" alt="Estructura de la red generadora" width="800"/>

Esta red estará formada por 11 capas de las cuales 3 de ellas se corresponde con una función de activación y para su construcción crearemos una función que llamaremos en nuestra clase Model. A continuación se describen las diferentes capas:

* Capa Full-Connected (Dense)
* Capa Reshape (Reshape)
* Cada de convolución de dos dimensiones (Conv2D)
* Capa de despliegue de la función de activación ReLu (LeakyReLU)
* Capa de Dropout (Dropout)
* Capa de aplanamiento (Flatten)

Para la definición del proceso de creación de la red de neuronas de discrimización (discriminator) debemos crear una función denominada __build_discriminator_model__ que tendrán tres parámetros de entrada:

* name (string): Este parámetro se corresponde con el identificador del modelo. 
* height (int): Este parámetro se corresponde con la altura de la imagen. 
* width (int): Este parámetro se corresponde con la anchura de la imagen.
* dims (int): Se corresponde con el número de canales de la imagen. En este taller vamos a trabajar con imágenes en blanco y negro (monocromo) por lo que sólo tendremos un canal. 

```
def build_discriminator_model(name, height, width, dims):
    
    model = Sequential()
    
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[height, width, dims]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model
```

Como se puede observar esta red de neuronas es una red de neurona de tipo convolucional clásica para la identificación de la clase. Esta red será la que se utilizará para predecir si los ejemplos (imágenes) generados por nuestra red generadora con correcto o no enviado su feedback a esta red con el objetivo de mejorar el proceso de aprendizaje. 

**Paso 7: Definición de la clase Model**

Para simplificar la creación de nuestro modelo, vamos a construir una clase para su manipulación. Para ello crearemos una clase en python denominada Model que tendrá ocho atributos:

* Red generadora (generator) que almacenará el modelo construido para la generación de los datos durante el proceso de entrenamiento.
* Red discriminadora (Discriminator) que almacenará el modelo construido para la eveluación de los datos en el proceso de entrenamiento. 
* Algoritmo de optimización del generador (generator_optimizer) que almacenará el algoritmo de optimización que será utilizado durante el proceso de entrenamiento de la red generadora. En este caso se utilizará el algoritmo de [Adam](https://arxiv.org/abs/1412.6980).
* Algoritmo de optimización del discriminador (discriminator_optimizer) que almacenará el algoritmo de optimización que será utilizado durante el proceso de entrenamiento de la red discriminadora. En este caso se utilizará el algoritmo de [Adam](https://arxiv.org/abs/1412.6980).
* Directorio de los punto de guardado (checkpoint_dir) que almacenará la ruta donde se almacenan las copias de seguridad de las redes durante el proceso de entrenamiento. 
* Prefijo para el nombrado de los fichero de punto de guardado (checkpoint_prefix) que almacenará el nombre que se incluirá en todos los fichero de tipo checkpoint. 
* El objeto para la creación de los puntos de guardado (checkpoint) que almacenará el objeto que permitirá crear los diferentes punto de guardado durante el proceso de entrenamiento. 

```
class Model:

  def __init__(self, height, width, dims):
  
    self.__generator = build_generator_model('generator_test', height, width, dims)
    self.__discriminator = build_discriminator_model('discriminator_test', height, width, dims)
    self.__cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    self.__generator_optimizer = optimizers.Adam(1e-4)
    self.__discriminator_optimizer = optimizers.Adam(1e-4)

    self.__checkpoint_dir = './training_checkpoints'
    self.__checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    self.__checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=self.__generator,
                                 discriminator=self.__discriminator)
    
    self.__checkpoint.restore(tf.train.latest_checkpoint(self.__checkpoint_dir))
```

Al final del proceso de creación de las variables locales del modelo, el sistema comprobará si existen algún checkpoint previo con el objetivo de continuación el proceso de entrenamiento desde ahí. A continuación es necesario crear una serie de método para manipular los diferentes elementos del modelo:

El primer método o función se corresponde con la visualización de la estructura de la red de neuronas generadora que denominaremos __generator_summary__. 

```
  def generator_summary(self):
    return self.__generator.summary()
```
El segundo método o función se corresponde con la visualización de la estructura de la red de neuronas discriminadora que denominaremos __generator_summary__. 

```
  def discriminator_summary(self):
    return self.__discriminator.summary()   
```

El tercer método se corresponde con una propiedad que nos devolverá el modelo generador con el objetivo de utilizarlo. Este método se denominará __generator__. 

```
  @property
  def generator(self):
    return self.__generator
```

El cuarto método se corresponde con una propiedad que nos devolverá el modelo discriminador con el objetivo de utilizarlo. Este método se denominará __discriminator__. 

```
  @property
  def discriminator(self):
    return self.__discriminator
```

El quinto método se corresponde con una propiedad que nos devolverá el algoritmo de optimización utilizado para la red generadora. Este método se denominará __generator_optimizer__. 

```
  @property
  def generator_optimizer(self):
    return self.__generator_optimizer
```

El sexto método se corresponde con una propiedad que nos devolverá el algoritmo de optimización utilizado para la red discriminadora. Este método se denominará __discriminator_optimizer__. 

```
  @property
  def discriminator_optimizer(self):
    return self.__discriminator_optimizer 
```

El septimo método o función se corresponde con la función de los utilizada por la red discriminadora que denominaremos __discriminator_loss__. Esta función utilizará dos parámetros:

* real_output que se corresponde con el valor real al que se corresponde el valor de entrada (clase). 
* fake_output que se corresponde con el valor tras la realización del proceso de inferencia sobre la red discriminadora (clase). 

Esta función devolverá un valor entre 0 y 2.

```
  @tf.autograph.experimental.do_not_convert
  def discriminator_loss(self, real_output, fake_output):
    real_loss = self.__cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = self.__cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss
```

El octavo método o función se corresponde con la función de los utilizada por la red generadora que denominaremos __generator_loss__. Esta función utilizará un único parámetros:

* fake_output que se corresponde con el valor que se corresponde con el ejemplo generado por la red generadora. 

Esta función devolverá un valor entre 0 y 2.

```
  @tf.autograph.experimental.do_not_convert
  def generator_loss(self, fake_output):
    return self.__cross_entropy(tf.ones_like(fake_output), fake_output)
```
El noveno método o función se corresponde con la función dque generá los punto de guardado de la red durante el proceso de entrenamiento y  que denominaremos __create_checkpoint__.

```
  def create_checkpoint(self):
    self.__checkpoint.save(file_prefix = self.__checkpoint_prefix)
```

**Paso 8 - Definición de la función de generación de ejemplos**

Una vez que hemos definido nuestro modelo tenemos que crear las diferentes funciones del proceso de entrenamiento. Para ello crearemos una función denominada __train_step__ de tipo tf.function (). Esta función recibirá 6 parámetros de entrada:

* model (Model) se corresponde con el modelo que está siendo entrenado.
* writer (TF Writer) se corresponde con el objeto utilizado para almacena la información para evaluación el proceso de entrenamiento. 
* images () se corresponde con las imágenes reales utilizadas para entrenar a la red discriminadora. 
* num_examples (int) se corresponde con el numero de imagenes que serán generado en pasa iteración de entrenamiento. 
* example_shape (tupple(int, int)) se corresponde con la estructura de la imagenes. 
* epoch (int) se corresponde con la iteración actual del proceso de entrenamiento. 

```
@tf.function
def train_step(model, writer, images, example_num, example_shape, epoch):
    
    examples = tf.random.normal([example_num, example_shape])
    
    with writer.as_default():
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = model.generator(examples, training=True)
        real_output = model.discriminator(images, training=True)
        fake_output = model.discriminator(generated_images, training=True)
        
        gen_loss = model.generator_loss(fake_output)
        disc_loss = model.discriminator_loss(real_output, fake_output)

      tf.summary.scalar("generator loss", gen_loss, step=epoch+1)
      tf.summary.scalar("discriminator loss", disc_loss, step=epoch+1)

      gradients_of_generator = gen_tape.gradient(gen_loss, model.generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, model.discriminator.trainable_variables)

      model.generator_optimizer.apply_gradients(zip(gradients_of_generator, model.generator.trainable_variables))
      model.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, model.discriminator.trainable_variables))
```

Además crearemos una función denominada ____call____ que nos permitirá calcular el valor de y para cada una de las x en base a los valores calculados. 

```
    def __call__(self, x):
      return tf.linalg.matmul(x, self.weights) + self.bias
```

y una función denominada add que nos permitirá añadir una nuevo valor a las diferentes métricas almacenadas en vars indicando el nombre de la métrica y el valor. 

```
    def add(self, variable, value):
      self.vars[variable].append(value)
```

A continuación tenemos que cargar los datos en las estructuras de datos básicas para comenzar a trabajar con ellos. Por lo que necesitaremos dos conjuntos de datos:

* Ejemplos (entrenamiento/validacion/test): Conjunto de ejemplos de información (imágenes) para los procesos de entrenamiento, validación y test.
* Labels (clases): Conjunto de clases asignadas a cada una de las imágenes de los diferentes conjuntos. Cada conjunto tendrá un conjunto de labels del mismo tamaño. 

Para poder cargar los datos en formato minist tenemos que utilizar las funcionalidades de importación propuesta por el equipo de TensorFlow (version 2016). Para ello debemos cargar el código de las funciones de cargar mediante la inclusión de un archivo local que denominaremos __input_data.py__. El código fuente de este archivo, se puede descargar en la siguiente [url](./resources/exercise_4/input_data.py). Una vez incluido este archivo podemos realizar la carga de datos. Para ellos utilizaremos la función __read_data_sets__ que nos permite cargar dataset desde una url, utilizando las siguiente opciones:

- Nombre del dataset
- source_url: Se corresponde con la url donde estará almacenada la información. 
- one_hot:  Realiza una transformación sobre las variables categorizadas a una codificación binaria. Es decir si tenemos n valores para una variables categorica se crearan n features binarias (0,1) de forma que sólo una de ellas tendrá el valor 1 correspondiendose con uno de los valores de la variable categorica. En este ejercicio, se utiliza para convertar la caraterística label (Clase de salida) en una coficiación binaria. 

```
full_data = input_data.read_data_sets('data', one_hot=True)

# Condificación one hot para un ejemplo de tipo Dress
# 3 => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# El valor tres se convierte en una array de probabilidades donde la posición que se corresponde con la etiqueta (3) tiene una probabilidad de 1 y el resto tienen una probabilidad de cero.


LABELS = {
 0: 'Camiseta/top',
 1: 'Pantalones',
 2: 'Sudadera',
 3: 'Vestido',
 4: 'Abrigo/Gabardina',
 5: 'Sandalias/Zapato',
 6: 'Camisa',
 7: 'Zapatillas',
 8: 'Bolso/Bolsa',
 9: 'Botas',
}
```

**Paso 5: Análisis de datos**

Una vez que hemos cargado los datos, tenemos que analizar los datos para entender su estructura, formato, si tenemos disponibles los suficientes conjuntos de entrenamiento, etc. Para ellos vamos a analizar algunas caracteristicas de los datos. Primero comprobaremos el tamaño (shape) de los conjuntos de datos que vamos a utilizar:

```
print("Conjunto de entrenamiento (Imágenes) shape: {shape}".format(shape=full_data.train.images.shape))
print("Conjunto de entrenamiento (Classes) shape: {shape}".format(shape=full_data.train.labels.shape))
print("Conjunto de test (Imágenes) shape: {shape}".format(shape=full_data.test.images.shape))
print("Conjunto de test (Clases) shape: {shape}".format(shape=full_data.test.labels.shape))
```

Cómo podemos observar tenemos 55000 ejemplos (imágenes) de entrenamiento donde cada uno de ellas está formada por un array de 784 pixeles y pueden pertener a 10 clases diferentes. 

Lo primero a analizar, es comocer si el número de clases y la estructura de los ejemplos es similar tanto en el conjunto de test como entrenamiento. En este caso ambos conjuntos están bien formados, tenemos un conjunto global formado por 65000 ejemplos donde un 18% se corresponde con el conjunto de test y todas la imágenes tienen el mismo tamaño (784 pixeles). 

Lo segundo a comprobar es el formato de la imagen, tenemos que definir cual es la estructura de la imagen. Para comprobarlo, bastaría con calcular la raíz cuadrada de 784 que se corresponde con 28. Esto significa que las imágenes tiene un tamaño de 28x28 pixeles. __Es obligatorio que todas las imágenes tenga el mismo tamaño, sino no podremos construir nuestra red de neuronas__. 

```
print(full_data.train.images[0].shape)
print(math.sqrt(full_data.train.images[0].shape[0]))
image_size = int(math.sqrt(full_data.train.images[0].shape[0]))
print(image_size)

print(full_data.train.labels[0])
```

Lo tercero es comprobar el volumen de ejemplos de cada clase en el conjunto de entrenamiento y en el conjunto de test. Normalmente nosotros tendremos que crear estos conjuntos, por lo que es muy útil comprobar si los conjuntos están balanceados, con el objetivo de balancearlos en caso de que no ocurra. 

```
train_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
test_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for label in full_data.train.labels:
    train_labels = np.add(train_labels, label)
    
print(train_labels)

for label in full_data.test.labels:
    test_labels = np.add(test_labels, label)

print(test_labels)
```

**Paso 6: Visualización de los datos**

Por último vamos a crear una función para visualizar las imágenes con las que estamos trabajando con el objetivo de ver el tipo de imágenes que estamos utilizando. La función se denominará __plot_image__ y nos permitirá visualizar imágenes con su etiqueta y utilizará 5 parámetros de entrada:

- plt: Es la figura sobre la que se insertará la imagen. 
- data: Se corresponde con la imagen que queremos visualizar. 
- label: Se corresponde con la etiqueta asignada a la imagen. Es un vector de n valores. 
- size: El tamaño de la imagen. Es una tupla con dos valores. 
- location: Es la localización de la imagen en la figura. Se corresponde con un secuencia de tres número enteros. 

```
def plot_image(plt, data, label, size, location):
    plt.subplot(location)
    img = np.reshape(data, size)
    label = np.argmax(label)
    plt.imshow(img)
    plt.title("(Label: " + str(LABELS[label]) + ")")
```

Una vez que hemos generado la función para visualizar la estructura de los ejemplos y las etiquetas (labels) podemos utilizarla para mostrar algunos de nuestros ejemplos mediante el siguiente fragmento de código:

```
plt.figure(figsize=[18,18])

plot_image(plt, 
           full_data.train.images[4], 
           full_data.train.labels[4,:], 
           (image_size, image_size),
           121)

plot_image(plt, 
           full_data.test.images[95], 
           full_data.test.labels[95,:], 
           (image_size, image_size),
           122)
```

**Congratulations Ninja!**

Has aprendido como preparar los datos para el proceso de aprendizaje, como definir las clases del modelo y como distribuir los conjunto de entrenamiento y test. Has conseguido aprender:

1. Como instalar paquetes en un notebook. 
2. Como descargar archivo mediante python request.
3. Como definir las clases o etiquetas (label) para la construcción de un modelo de aprendizaje. 
4. Como realizar un análisis básico sobre los datos de entrenamamiento y test.
5. Como visualizar imágenes mediante matplotlib. 

<img src="../img/ejercicio_3_congrats.png" alt="Congrats ejercicio 3" width="800"/>


