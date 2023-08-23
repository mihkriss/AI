[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10795754&assignment_repo_type=AssignmentRepo)
# Лабораторная работа по курсу "Искусственный интеллект"
# Создание своего нейросетевого фреймворка

### Студенты: 

| ФИО       | Роль в проекте                     | Оценка       |
|-----------|------------------------------------|--------------|
| Варламова А.Б. | Написание фреймворк для обучения, написание функций обучения и перемешивания датасета, написание примеров, тестирование, обучение нейросети, исправление ошибок |          |
| Михеева К.О. | Написание функции методов оптимизации(SGD, Momentum SGD, GradientClipping), функции ошибок для классификации(CrossEntropyLoss, BinaryCrossEntropy) и регрессии(MeanSquaredError, AbsoluteError), передаточные функции(Softmax, Sigmoid, Threshold, Tanh), тестирование, обучение нейросети |    |
| Пономарев Н.В.   | Писал отчёт |          |

> *Комментарии проверяющего*

### Задание

Реализовать свой нейросетевой фреймворк для обучения полносвязных нейросетей, который должен включать в себя следующие возможности:

1. Создание многослойной нейросети перечислением слоёв
1. Удобный набор функций для работы с данными и датасетами (map, minibatching, перемешивание и др.)
1. Несколько (не менее 3) алгоритмов оптимизации: SGD, Momentum SGD, Gradient Clipping и др.
1. Описание нескольких передаточных функций и функций потерь для решения задач классификации и регрессии.
1. Обучение нейросети "в несколько строк", при этом с гибкой возможностью конфигурирования
1. 2-3 примера использования нейросети на классических задачах (MNIST, Iris и др.)
1. Документация в виде файла README.md 

> Документацию можно писать прямо в этом файле, удалив задание, но оставив табличку с фамилиями студентов. Необходимо разместить решение в репозитории Github Classroom, но крайне рекомендую также сделать для этого отдельный открытый github-репозиторий (можно путём fork-инга).

### Описание фреймворка

Классы "SGD", "MomentumSGD" и "GradientClipping" отвечают за различные алгоритмы оптимизации, которые используются для обновления весов нейронной сети в процессе обучения.

#### Класс "SGD" 
Реализует стохастический градиентный спуск (Stochastic Gradient Descent). Это один из самых популярных и простых алгоритмов оптимизации, который используется в машинном обучении. Он заключается в том, что на каждом шаге обучения мы выбираем случайный набор данных (обычно размером с мини-батч) и обновляем веса, используя градиент функции потерь по этому набору данных. Это позволяет снизить вычислительную сложность и использовать большие наборы данных для обучения нейронной сети. Класс "SGD" реализует стохастический градиентный спуск, который является одним из самых простых алгоритмов оптимизации. Он обновляет веса сети по формуле:
```py
new_weight = old_weight - learning_rate * gradient
```
Где learning_rate - скорость обучения, gradient - градиент функции ошибки по весам.


В случае класса "SGD" learning rate следует выбирать достаточно маленьким, например, 0.01 или меньше, чтобы избежать осцилляций и быстрого расхождения при обновлении весов.

#### Класс "MomentumSGD" 
Реализует градиентный спуск с моментом (Momentum Gradient Descent). Этот алгоритм оптимизации также использует градиент функции потерь для обновления весов нейронной сети, но с учетом предыдущих изменений весов. Он добавляет инерцию к градиенту, чтобы сгладить колебания и ускорить сходимость алгоритма оптимизации. Это достигается путем добавления вектора момента, который вычисляется как взвешенная сумма предыдущего вектора момента и текущего градиента. Класс "MomentumSGD" реализует градиентный спуск с моментом, который помогает преодолеть проблему медленной сходимости в плоских областях и осцилляций в ущельях. Он обновляет веса сети по формуле:
```py
new_weight = old_weight - learning_rate * gradient + momentum * (old_weight - prev_weight)
```
Где learning_rate - скорость обучения, gradient - градиент функции ошибки по весам, momentum - коэффициент момента, prev_weight - предыдущее значение весов.

В случае класса "MomentumSGD" learning rate можно выбирать больше, например, 0.1 или 0.01, чтобы ускорить сходимость и получить лучший результат.

#### Класс "GradientClipping"
Реализует ограничение градиента (Gradient Clipping). Этот алгоритм оптимизации используется для предотвращения "взрыва градиента" в глубоких нейронных сетях. В некоторых случаях градиент может становиться очень большим или очень маленьким, что может замедлить или полностью остановить обучение нейронной сети. Для решения этой проблемы мы можем ограничить градиенты по модулю, используя заданный порог. Это позволяет нам сохранить общее направление обновления весов, но предотвратить неожиданные большие изменения. Класс "GradientClipping" реализует ограничение градиента, которое помогает справиться с проблемой взрыва градиента в глубоких сетях. Он обновляет веса сети по формуле:
```py
new_weight = old_weight - learning_rate * clipped_gradient
```
Где learning_rate - скорость обучения, clipped_gradient - ограниченный градиент.

В случае класса "GradientClipping" learning rate следует выбирать достаточно маленьким, например, 0.01 или меньше, чтобы избежать быстрого расхождения при обновлении весов.

#### Классы "CrossEntropyLoss", "BinaryCrossEntropy", "MeanSquaredError" и "AbsoluteError" 
Отвечают за функции ошибки. Класс "CrossEntropyLoss" реализует кросс-энтропию для задач классификации, "BinaryCrossEntropy" - бинарную кросс-энтропию для задач бинарной классификации, "MeanSquaredError" - среднеквадратичную ошибку для задач регрессии, а "AbsoluteError" - среднюю абсолютную ошибку для задач регрессии.

#### Класс "NeuralNetwork" 
Содержит методы для обучения нейронной сети. Метод "train()" выполняет обучение сети на заданном наборе данных, метод "predict()" выполняет предсказание на новых данных, а метод "test()" тестирует сеть на заданном наборе данных и вычисляет метрики качества, метод "create()", который инициализирует атрибуты класса, включая количество слоев, количество нейронов в каждом слое, функцию активации для каждого слоя и методы для обучения и использования нейронной сети.

#### Функции "get_loss_acc()" и "my_shuffle()" 
Определены вне классов и используются внутри методов класса "NeuralNetwork". Функция "get_loss_acc()" вычисляет значение функции ошибки и точности на заданном наборе данных, а функция "my_shuffle()" выполняет перемешивание набора данных.

### Примеры использования фреймворка
#### MNIST
В этом коде загружается набор данных MNIST с помощью библиотеки Keras и делится на тренировочный и тестовый наборы.
Далее производится предобработка данных. В частности, обучающая выборка ограничивается 1000 примерами, тестовая - 200. Далее значения пикселей изображений нормируются на диапазон [0,1], а также преобразуются в одномерные массивы.
```py
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train[0:1000]
Y_train = Y_train[0:1000]

X_test = X_test[0:200]
Y_test = Y_test[0:200]


X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

train_x = np.array(X_train)
test_x = np.array(X_test)
n_train = train_x.shape[0]
n_test = X_test.shape[0]
train_x = np.reshape(train_x, (n_train, 784))
test_x = np.reshape(X_test, (n_test, 784))
```
- Обучение
```py
mnist_model = NeuralNetwork()
mnist_model.create(layers = [Linear(784, 400), Threshold(), Linear(400, 80), Tanh(), Linear(80, 10), Softmax()],
                 lossFunc = CrossEntropyLoss(), optim=SGD(), epochsNumber=15, learning_rate=0.09, minibatch=100)
mnist_model.fit(train_x, Y_train)
pred_vals, acc = mnist_model.test(test_x, Y_test)

print(f'Test accuracy = {acc}')
```
- Вывод
```
Epoch  1 :
Loss=0.4911070090725692, accuracy=0.80375: 

Epoch  2 :
Loss=0.4738841975863627, accuracy=0.80625: 

Epoch  3 :
Loss=0.46587150131113153, accuracy=0.81: 

Epoch  4 :
Loss=0.46168967730917193, accuracy=0.80375: 

Epoch  5 :
Loss=0.4585265054853119, accuracy=0.80375: 

Epoch  6 :
Loss=0.4552111397675765, accuracy=0.80875: 

Epoch  7 :
Loss=0.4520009637212202, accuracy=0.81625: 

Epoch  8 :
Loss=0.4492642923582158, accuracy=0.8225: 

Epoch  9 :
Loss=0.4469224584554722, accuracy=0.8225: 

Epoch  10 :
Loss=0.4448453556757583, accuracy=0.8225: 

Epoch  11 :
Loss=0.44297118337026875, accuracy=0.8225: 

Epoch  12 :
Loss=0.4412656528231808, accuracy=0.8225: 

Epoch  13 :
Loss=0.4397010687070466, accuracy=0.81875: 

Epoch  14 :
Loss=0.4382392352332603, accuracy=0.815: 

Epoch  15 :
Loss=0.4367908154109755, accuracy=0.81875: 

Epoch  16 :
Loss=0.43508339700034765, accuracy=0.83: 

Epoch  17 :
Loss=0.4327415551487921, accuracy=0.8375: 

Epoch  18 :
Loss=0.4307443129514752, accuracy=0.8375: 

Epoch  19 :
Loss=0.4297090510712894, accuracy=0.8375: 

Epoch  20 :
Loss=0.42887269312939524, accuracy=0.8375: 

Test accuracy: 0.835
```

#### Классификация
В этом коде генерируется набор данных, используя функцию make_classification из библиотеки scikit-learn. Для этого создается 1000 образцов с 2 признаками и различными параметрами. После этого данные разбиваются на обучающую и тестовую выборки в соотношении 4:1.
```py
n = 1000
X, Y = make_classification(n_samples = n, n_features=2,
                           n_redundant=0, n_informative=2, flip_y=0.2)
X = X.astype(np.float32)
Y = Y.astype(np.int32)

# Разбиваем на обучающую и тестовые выборки
train_x, test_x = np.split(X, [n*8//10])
train_labels, test_labels = np.split(Y, [n*8//10])
```
- Обучение
```py
model = NeuralNetwork()
model.create(layers=[Linear(2,5), Tanh(), Linear(5,2), Softmax()], lossFunc=CrossEntropyLoss(), optim=SGD(), epochsNumber=20, learning_rate=0.2, minibatch=4)
model.fit(train_x,train_labels)
loss, acc = model.test(test_x, test_labels)
print(f'Test: {acc}')
```
- Визуализация датасета
```py
def plot_dataset(suptitle, features, labels):
    # prepare the plot
    fig, ax = plt.subplots(1, 1)
    #pylab.subplots_adjust(bottom=0.2, wspace=0.4)
    fig.suptitle(suptitle, fontsize = 16)
    ax.set_xlabel('$x_i[0]$ -- (feature 1)')
    ax.set_ylabel('$x_i[1]$ -- (feature 2)')

    colors = ['r' if l else 'b' for l in labels]
    ax.scatter(features[:, 0], features[:, 1], marker='o', c=colors, s=100, alpha = 0.5)
    plt.show()
    plot_dataset('Scatterplot of the training data', train_x, train_labels)
```
![image](https://user-images.githubusercontent.com/71276784/236262737-8a51a088-45a4-43ad-9e9a-d42efde37691.png)

- Вывод
```
Epoch  1 :
Loss=0.4911070090725692, accuracy=0.80375: 

Epoch  2 :
Loss=0.4738841975863627, accuracy=0.80625: 

Epoch  3 :
Loss=0.46587150131113153, accuracy=0.81: 

Epoch  4 :
Loss=0.46168967730917193, accuracy=0.80375: 

Epoch  5 :
Loss=0.4585265054853119, accuracy=0.80375: 

Epoch  6 :
Loss=0.4552111397675765, accuracy=0.80875: 

Epoch  7 :
Loss=0.4520009637212202, accuracy=0.81625: 

Epoch  8 :
Loss=0.4492642923582158, accuracy=0.8225: 

Epoch  9 :
Loss=0.4469224584554722, accuracy=0.8225: 

Epoch  10 :
Loss=0.4448453556757583, accuracy=0.8225: 

Epoch  11 :
Loss=0.44297118337026875, accuracy=0.8225: 

Epoch  12 :
Loss=0.4412656528231808, accuracy=0.8225: 

Epoch  13 :
Loss=0.4397010687070466, accuracy=0.81875: 

Epoch  14 :
Loss=0.4382392352332603, accuracy=0.815: 

Epoch  15 :
Loss=0.4367908154109755, accuracy=0.81875: 

Epoch  16 :
Loss=0.43508339700034765, accuracy=0.83: 

Epoch  17 :
Loss=0.4327415551487921, accuracy=0.8375: 

Epoch  18 :
Loss=0.4307443129514752, accuracy=0.8375: 

Epoch  19 :
Loss=0.4297090510712894, accuracy=0.8375: 

Epoch  20 :
Loss=0.42887269312939524, accuracy=0.8375: 

Test: 0.835
```

#### Регрессия
Здесь создаются данные для задачи регрессии с помощью функции make_regression. Генерируются n данных с одним признаком, одним информативным признаком, шумом и заданным случайным состоянием. Затем данные разбиваются на обучающую и тестовую выборки с соотношением 8:2 и для обучающей выборки производится решейпинг меток классов.
```py
n = 100
X, Y = make_regression(n_samples=n, n_features=1, 
                       n_informative=1, noise = 10, random_state=0)

# Разбиваем на обучающую и тестовые выборки
train_x, test_x = np.split(X, [n*8//10])
train_y, test_y = np.split(Y, [n*8//10])
train_y = np.reshape(train_y, (80, 1))
```
- Обучение
```py
model = NeuralNetwork()
model.create(layers=[Linear(1,1)], lossFunc=AbsoluteError(), optim=SGD(), epochsNumber=15, learning_rate=0.1, minibatch=10)
model.fit(train_x,train_y)
loss = model.test(test_x,test_y)
print(f'Тест: {loss}')
```
- Вывод
```
Epoch  1 :
Loss=47.81146821732139: 

Epoch  2 :
Loss=46.81994557655644: 

Epoch  3 :
Loss=45.835229950746694: 

Epoch  4 :
Loss=44.85051432493694: 

Epoch  5 :
Loss=43.86579869912718: 

Epoch  6 :
Loss=42.881083073317434: 

Epoch  7 :
Loss=41.89636744750768: 

Epoch  8 :
Loss=40.91165182169793: 

Epoch  9 :
Loss=39.926936195888175: 

Epoch  10 :
Loss=38.942220570078426: 

Epoch  11 :
Loss=37.95750494426868: 

Epoch  12 :
Loss=36.972789318458936: 

Epoch  13 :
Loss=35.98807369264917: 

Epoch  14 :
Loss=35.003358066839425: 

Epoch  15 :
Loss=34.01864244102968: 

Тест: 24.867631121885292
```
#### Ирисы
Здесь происходит загрузка датасета iris, который содержит информацию о трех видах ирисов. Затем данные и метки разбиваются на обучающую и тестовую выборки в соотношении 4:1. Перед этим данные и метки перемешиваются с помощью функции my_shuffle(), чтобы обеспечить случайность разбиения и улучшить качество обучения модели.
```py
iris = load_iris()
X, Y = iris.data[:, :3], iris.target
n = X.shape[0]
my_shuffle(X, Y)
# Разбиваем на обучающую и тестовые выборки
train_x, test_x = np.split(X, [n*8//10])
train_labels, test_labels = np.split(Y, [n*8//10])
```
- Обучение
```py
iris_model = NeuralNetwork()
iris_model.create(layers = [Linear(3, 3), Threshold(), Linear(3, 3), Tanh(), Linear(3, 3), Softmax()],
                 lossFunc = CrossEntropyLoss(), optim = SGD(), epochsNumber=30, learning_rate = 0.1, minibatch=5)
iris_model.fit(train_x, train_labels)
pred_vals, acc = iris_model.test(test_x, test_labels)

print(f'Test accuracy = {acc}')
```
- Вывод
```
Epoch  1 :
Loss=0.8803576948107769, accuracy=0.6333333333333333: 

Epoch  2 :
Loss=0.8625178796473764, accuracy=0.6333333333333333: 

Epoch  3 :
Loss=0.8589304901000001, accuracy=0.6333333333333333: 

Epoch  4 :
Loss=0.8584458404635311, accuracy=0.6333333333333333: 

Epoch  5 :
Loss=0.8597702978448917, accuracy=0.6333333333333333: 

Epoch  6 :
Loss=0.8621950240317925, accuracy=0.6333333333333333: 

Epoch  7 :
Loss=0.8656634683775418, accuracy=0.6333333333333333: 

Epoch  8 :
Loss=0.7803726940264465, accuracy=0.6333333333333333: 

Epoch  9 :
Loss=0.6311497447381703, accuracy=0.6333333333333333: 

Epoch  10 :
Loss=0.7475751071249223, accuracy=0.6333333333333333: 

Epoch  11 :
Loss=0.6176458985590166, accuracy=0.8: 

Epoch  12 :
Loss=0.44916174363987166, accuracy=0.8833333333333333: 

Epoch  13 :
Loss=0.39981253642669295, accuracy=0.9: 

Epoch  14 :
Loss=0.43804910173443606, accuracy=0.875: 

Epoch  15 :
Loss=0.5457366672111152, accuracy=0.8: 

Epoch  16 :
Loss=0.5432723939500804, accuracy=0.8: 

Epoch  17 :
Loss=0.6222833465445636, accuracy=0.7166666666666667: 

Epoch  18 :
Loss=0.6290345985889911, accuracy=0.7166666666666667: 

Epoch  19 :
Loss=0.6268724963117164, accuracy=0.7166666666666667: 

Epoch  20 :
Loss=0.6362022495475645, accuracy=0.7166666666666667: 

Epoch  21 :
Loss=0.6269960666035873, accuracy=0.7166666666666667: 

Epoch  22 :
Loss=0.608874900805697, accuracy=0.7166666666666667: 

Epoch  23 :
Loss=0.6023591750697946, accuracy=0.7166666666666667: 

Epoch  24 :
Loss=0.5973965121619808, accuracy=0.7166666666666667: 

Epoch  25 :
Loss=0.5936087501768975, accuracy=0.7166666666666667: 

Epoch  26 :
Loss=0.5906683556385296, accuracy=0.7166666666666667: 

Epoch  27 :
Loss=0.5930420884907673, accuracy=0.7166666666666667: 

Epoch  28 :
Loss=0.5905876386349378, accuracy=0.7166666666666667: 

Epoch  29 :
Loss=0.5885777654208657, accuracy=0.7166666666666667: 

Epoch  30 :
Loss=0.5869113474165725, accuracy=0.7166666666666667: 

Test accuracy = 0.7333333333333333
```
