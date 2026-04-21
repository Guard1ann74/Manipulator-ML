# Журнал по разработке системы распознавания команд для манипулятора

**02.04.2026**

Взял наработки по манипулятору с другого проекта, чтобы разобраться в принципе работы и управления манипулятором. Несколько раз редактировал код, чтобы понять, как им управлять, и проверил, есть ли контроль силы сжатия захвата — контроль отсутствует, значит степень сжатия придется контролировать вручную. Манипулятор и захват представляют из себя две независимые системы, связь между которыми осуществляется через соединение по кабелю. Работа ведется с ноутбука, подключенного к манипулятору через сеть Wi-Fi (или через кабель Ethernet) и HDMI-кабель для захвата.

## Постановка задачи

Разработать модель для распознавания естественно-языковых команд управления манипулятором. На вход подаётся текстовая фраза, на выходе система должна определить, соответствует ли она одной из двух команд: «подняться вверх» - класс ```up``` или «опуститься вниз» класс ```down```. Все остальные фразы относятся к классу ```other``` и должны вызывать ответ «Действие не распознано».

## Сбор и подготовка данных

Созданы три текстовых файла в папке data/:

- **up.txt** — фразы, означающие поднятие вверх («поднимись вверх», «иди выше», «наверх»). Для начала решил взять 9 предложений, введенные вручную;
- **down.txt** — фразы, означающие опускание вниз («опустись вниз», «спустись», «вниз»). Абсолютно тот же объем;
- **other.txt** — фразы, не являющиеся командами («стоп», «поверни налево», «какая погода»). Вот тут я взял фраз больше, а именно 11 штук.

**Почему именно три файла?** Для обучения с учителем необходимы примеры всех классов. Без отрицательных примеров ```other``` модель не сможет отличать команды от посторонних фраз. Чем плох подход без этого класса - если мы введем предложение, вообще не совпадающее по смыслу, то модельке все равно придется выбрать один из двух классов. Если я введу например «пошли гулять», то она отнесет эту фразу либо к классу ```down```, либо к классу ```up```, что в свою очередь не является правдой.

## Выбор метода

### 1. Векторизация текста — TF-IDF (Term Frequency — Inverse Document Frequency)

**Что он делает:** Преобразует каждую фразу в числовой вектор, где каждый элемент соответствует важности конкретного слова.

**Почему выбран:** Это классический и эффективный метод для небольших и средних датасетов. Он не требует глубоких нейросетей, работает быстро на CPU и позволяет понять, какие слова влияют на решение.

**Параметры:** 
- `ngram_range=(1,2)` — учитываем униграммы и биграммы (например, «поднимись вверх»)
- `max_features=10000` — ограничиваем размер словаря, чтобы избежать переобучения

### 2. Классификатор — логистическая регрессия

**Что он делает:** Строит линейную разделяющую поверхность в пространстве признаков.

**Почему выбран:** Получаем вероятностный выход, что позволяет оценить уверенность в решении. При небольшом объёме данных работает стабильно и хорошо интерпретируется (веса признаков показывают вклад каждого слова).

## Методология оценки качества

**Разделение данных:** Обучающая выборка 80%, тестовая выборка 20% со стратификацией, чтобы сохранить пропорции классов в обеих частях.

>Стратификация — это метод разделения данных на выборки, при котором пропорции классов сохраняются одинаковыми.
> 
> Без нее может оказаться, что в тестовую выборку не попадёт ни одного слова «вниз» и мы не сможем оценить, как модель распознаёт это слово, потому что его нет в тесте.

**Метрики:**
- **Accuracy** - общая доля правильных ответов
- **Precision** - точность: из предсказанных положительных фраз сколько реально положительных
- **Recall** - полнота: из реальных положительных сколько найдено
- **F1-score** - гармоническое среднее Precision и Recall каждого класса, чтобы оценить, насколько хорошо модель распознаёт именно команды, не путая их с ```other``` и между собой
- **Матрица ошибок** — матрица, показывающая, какие классы чаще путаются.

## Сохранение модели

Обученные объекты записываются в файлы формата .pkl с помощью библиотеки joblib. Это позволяет загружать модель в любой момент без повторного обучения.

## Логика работы

1. Пользователь вводит фразу.
2. Векторизатор преобразует её в TF-IDF - вектор той же размерности, что и обучающие данные.
3. Классификатор вычисляет вероятности принадлежности к каждому из трёх классов.
4. Если максимальная вероятность соответствует классу ```up``` или ```down``` — выводится сообщение «Действие выполнено» и название команды. Если класс other — «Действие не распознано».

## Обоснование отказа от готовых LLM

Да просто не хотел тратить время на подбор подходящей модели и на дообучение + лучше самому разобраться, как это работает. Глядишь, переплюну Клод))

## Тестирование

**Тест 1. Ожидаемый результат**

После запуска сразу всплыла следующая ошибка: 
```
File "C:\Users\Holo\PycharmProjects\Manipularor + ML\model\train.py", line 38, in main
clf = LogisticRegression(max_iter=1000, multi_class='ovr', class_weight='balanced')
TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'
```
Проверил версию scikit-learn, версия последняя, значит проблема в другом. В коде есть одно предупреждение: *Unexpected argument* — `clf = LogisticRegression(max_iter=1000, multi_class='ovr', class_weight='balanced')`. Попробую убрать аргумент `multi_class`.

**Тест 2. Все заработало!** 

Обучение прошло успешно, в терминале получил следующий вывод
```
Загружено примеров: up=9, down=9, other=11

Оценка модели
Точность: 0.8333

Матрица ошибок:
[[2 0 0]
 [0 2 0]
 [1 0 1]]

Итоговые результаты:
              precision    recall  f1-score   support

       other       0.67      1.00      0.80         2
          up       1.00      1.00      1.00         2
        down       1.00      0.50      0.67         2

    accuracy                           0.83         6
   macro avg       0.89      0.83      0.82         6
weighted avg       0.89      0.83      0.82         6
 ```
Давайте-ка разберемся, что мы получили: датасет изначально был очень мал, тестовая выборка составила 20% от общего количества данных, то есть 29*20% = 6 примеров, с каждого класса по две фразы. Точность модели составила 0.833, что вполне норм. Разберем матрицу: 

#### Матрица ошибок

| Реально \ Предсказано | other | up | down |
|----------------------|-------|----| ----|
| other                | 2     | 0  | 0   |
| up                   | 0     | 2  | 0   |
| down                 | 1     | 0  | 1   |

Видим, что две фразы из класса ```other``` правильно предсказаны как ```other``` - это хорошо. Класс ```up``` так же не имеет ошибок, но вот в классе ```down``` видим, что одна фраза из этого класса ошибочно предсказана, как ```other```.
> Диагональ - правильные ответы и их количество

Значит, модель ошибается в 1 случае из 6. По таблице видим, что класс ```down``` подпортил нам статистику. Метрики, которые показаны ниже не сильно меня понятны, разберусь в них потом. 
## Результат
Посмотрим на выводы:
```
Доступные команды: 'вверх', 'вниз'
> тебе надо опуститься вниз
Действие выполнено: опуститься вниз | Точность: 58.50%
> тебе требуется вниз переместиться
Действие выполнено: опуститься вниз | Точность: 64.25%
> перемещайся ниже
Действие выполнено: опуститься вниз | Точность: 55.91%
> поднимайся выше
Действие выполнено: подняться вверх | Точность: 43.08%
> поднимись повыше
Действие выполнено: подняться вверх | Точность: 45.90%
> вверх иди
Действие выполнено: подняться вверх | Точность: 56.85%
> я хочу пиццы
Действие не распознано | Точность: 39.18%
```
В целом, результат положительный. Далее буду работать над сбором качественных данных, потому что 29 фраз для работы модели не хватит.

# **06.04.2026**

Решил отредачить датасет и добавить больше предложений с разными формулировками.

```
Загружено примеров: left=60, right=60, other=108

Оценка модели
Точность: 1.0000

Матрица ошибок:
[[22  0  0]
 [ 0 12  0]
 [ 0  0 12]]

Итоговые результаты:
              precision    recall  f1-score   support

       other       1.00      1.00      1.00        22
        left       1.00      1.00      1.00        12
       right       1.00      1.00      1.00        12

    accuracy                           1.00        46
   macro avg       1.00      1.00      1.00        46
weighted avg       1.00      1.00      1.00        46
```

Точность модели на минуточку составила 100%! Честно, меня это очень сильно усомнило. Протестим:
```
Система распознавания команд манипулятора
> тебе надо переместить кубик влево
Действие выполнено: переместить объект влево | Точность: 56.47%
> перемещай объект правее
Действие выполнено: переместить объект влево | Точность: 39.06%
> нужно сделать так, чтобы объект находился левее
Действие выполнено: переместить объект влево | Точность: 39.06%
> нужно сделать так, чтобы объект находился правее
Действие выполнено: переместить объект влево | Точность: 39.06%
> мдаа
Действие не распознано | Точность: 69.25%
```
Не все так радужно, как могло показаться. Несмотря на идеальнейшие результаты обучения, реальные тесты модель проигрывает. Попробую обогатить данные - добавить больше и разных формулировок. Теперь класс right и left содержат по 100 фраз, а класс other 158. Запускаем:

```
Загружено примеров: left=100, right=100, other=158

Оценка модели
Точность: 0.9444

Матрица ошибок:
[[29  2  1]
 [ 1 19  0]
 [ 0  0 20]]

Итоговые результаты:
              precision    recall  f1-score   support

       other       0.97      0.91      0.94        32
        left       0.90      0.95      0.93        20
       right       0.95      1.00      0.98        20

    accuracy                           0.94        72
   macro avg       0.94      0.95      0.95        72
weighted avg       0.95      0.94      0.94        72
```
```
Система распознавания команд манипулятора
> переставь кубик вправо
Действие выполнено: переместить объект вправо | Точность: 86.50%
> перемести предмет левее
Действие выполнено: переместить объект влево | Точность: 52.53%
> лево
Действие не распознано | Точность: 71.80%
> правее
Действие выполнено: переместить объект вправо | Точность: 84.45%
> вправо
Действие выполнено: переместить объект вправо | Точность: 89.33%
```
Пожалуй, стоит еще больше примеров добавить и несколько фраз "ловушек"

Новая табличка
```
Загружено примеров: left=120, right=120, other=178

Оценка модели
Точность: 0.9762

Матрица ошибок:
[[34  2  0]
 [ 0 24  0]
 [ 0  0 24]]

Итоговые результаты:
              precision    recall  f1-score   support

       other       1.00      0.94      0.97        36
        left       0.92      1.00      0.96        24
       right       1.00      1.00      1.00        24

    accuracy                           0.98        84
   macro avg       0.97      0.98      0.98        84
weighted avg       0.98      0.98      0.98        84
```
```
Система распознавания команд манипулятора
> тебе нужно переместить кубик правее
Действие выполнено: переместить объект вправо | Точность: 83.29%
> подвинь объект вниз
Действие не распознано | Точность: 44.18%
> подвинь объект влево
Действие выполнено: переместить объект влево | Точность: 68.91%
> подними предмет налево
Действие выполнено: переместить объект влево | Точность: 68.06%
```
Действие "Подними" было воспринято как действие "Влево", хотя это не так. Пока что проблема с датасетом не главная проблема, просто больше данных добавлю и все, может быть поковыряюсь в методе обучения ещё. Теперь, хочу наконец поиграть с манипулятором, буду использовать файли других ребят, которые уже до этого как-то работали с манипулятором.

```
C:\Users\Holo\PycharmProjects\Manipularor + ML>py -3.10 -m pip list
Package    Version
---------- -------
pip        21.2.3
setuptools 57.4.0
WARNING: You are using pip version 21.2.3; however, version 26.0.1 is available.
You should consider upgrading via the 'C:\Users\Holo\AppData\Local\Programs\Python\Python310\python.exe -m pip install --upgrade pip' command.

C:\Users\Holo\PycharmProjects\Manipularor + ML>py -3.11 -m pip list 
Package    Version
---------- -------
pip        24.0
setuptools 65.5.0

[notice] A new release of pip is available: 24.0 -> 26.0.1
[notice] To update, run: C:\Users\Holo\AppData\Local\Programs\Python\Python311\python.exe -m pip install --upgrade pip
```
К сожалению, версию питона пришлось откатить с версии 3.13 на 3.10, чтобы получилось полноценно управлять манипулятором. Также, когда я импортировал проект ```pyGhripper```, то не обратил внимание на то, что в нем тоже было свое виртуальное окружение, в котором было куча других нужных файлов. В итоге, пришлось очень много повозиться с этим.

```
C:\Users\Holo\PycharmProjects\Manipularor + ML>py -3.11 -m venv .venv
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\Holo\AppData\Local\Programs\Python\Python311\Lib\venv\__main__.py", line 6, in <module>
    main()
  File "C:\Users\Holo\AppData\Local\Programs\Python\Python311\Lib\venv\__init__.py", line 546, in main
    builder.create(d)
  File "C:\Users\Holo\AppData\Local\Programs\Python\Python311\Lib\venv\__init__.py", line 76, in create
    self._setup_pip(context)
  File "C:\Users\Holo\AppData\Local\Programs\Python\Python311\Lib\venv\__init__.py", line 358, in _setup_pip
    self._call_new_python(context, '-m', 'ensurepip', '--upgrade',
  File "C:\Users\Holo\AppData\Local\Programs\Python\Python311\Lib\venv\__init__.py", line 354, in _call_new_python
    subprocess.check_output(args, **kwargs)
  File "C:\Users\Holo\AppData\Local\Programs\Python\Python311\Lib\subprocess.py", line 466, in check_output
    return run(*popenargs, stdout=PIPE, timeout=timeout, check=True,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Holo\AppData\Local\Programs\Python\Python311\Lib\subprocess.py", line 550, in run
    stdout, stderr = process.communicate(input, timeout=timeout)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Holo\AppData\Local\Programs\Python\Python311\Lib\subprocess.py", line 1196, in communicate
    stdout = self.stdout.read()
             ^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
^C
C:\Users\Holo\PycharmProjects\Manipularor + ML>py -3.10 -m venv .venv 

C:\Users\Holo\PycharmProjects\Manipularor + ML>.venv\Scripts\activate

(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>python --version  
Python 3.10.0

(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>pip install scikit-learn 
Collecting scikit-learn
  Downloading scikit_learn-1.7.2-cp310-cp310-win_amd64.whl (8.9 MB)
     |████████████████████████████████| 8.9 MB 6.4 MB/s
Collecting joblib>=1.2.0
  Downloading joblib-1.5.3-py3-none-any.whl (309 kB)
     |████████████████████████████████| 309 kB 6.4 MB/s
Collecting threadpoolctl>=3.1.0
  Using cached threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Collecting numpy>=1.22.0
  Downloading numpy-2.2.6-cp310-cp310-win_amd64.whl (12.9 MB)
     |████████████████████████████████| 12.9 MB 6.4 MB/s
Collecting scipy>=1.8.0
  Downloading scipy-1.15.3-cp310-cp310-win_amd64.whl (41.3 MB)
     |████████████████████████████████| 41.3 MB 3.3 MB/s
Installing collected packages: numpy, threadpoolctl, scipy, joblib, scikit-learn
Successfully installed joblib-1.5.3 numpy-2.2.6 scikit-learn-1.7.2 scipy-1.15.3 threadpoolctl-3.6.0
WARNING: You are using pip version 21.2.3; however, version 26.0.1 is available.
You should consider upgrading via the 'C:\Users\Holo\PycharmProjects\Manipularor + ML\.venv\Scripts\python.exe -m pip install --upgrade pip' command.
```
Теперь устанавливаем все нужные зависимости заново.
```
(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>pip install -r requirements.txt
Collecting protobuf==4.23.0
  Downloading protobuf-4.23.0-cp310-abi3-win_amd64.whl (422 kB)
     |████████████████████████████████| 422 kB 939 kB/s
Collecting protocol==0.1.0
  Downloading protocol-0.1.0-py2.py3-none-any.whl (3.3 kB)
Collecting requests~=2.28.2
  Downloading requests-2.28.2-py3-none-any.whl (62 kB)
     |████████████████████████████████| 62 kB 4.8 MB/s
Collecting google~=3.0.0
  Downloading google-3.0.0-py2.py3-none-any.whl (45 kB)
     |████████████████████████████████| 45 kB 1.3 MB/s
Collecting pynng~=0.7.2
  Downloading pynng-0.7.4-cp310-cp310-win_amd64.whl (425 kB)
     |████████████████████████████████| 425 kB 6.8 MB/s
Collecting cffi==1.15.1
  Downloading cffi-1.15.1-cp310-cp310-win_amd64.whl (179 kB)
     |████████████████████████████████| 179 kB 6.4 MB/s
Collecting APScheduler~=3.10.1
  Downloading APScheduler-3.10.4-py3-none-any.whl (59 kB)
     |████████████████████████████████| 59 kB 3.2 MB/s
Collecting setuptools~=67.6.1
  Downloading setuptools-67.6.1-py3-none-any.whl (1.1 MB)
     |████████████████████████████████| 1.1 MB ...
Collecting aenum~=3.1.12
  Downloading aenum-3.1.17-py3-none-any.whl (165 kB)
     |████████████████████████████████| 165 kB ...
Collecting pytest~=7.2.2
  Downloading pytest-7.2.2-py3-none-any.whl (317 kB)
     |████████████████████████████████| 317 kB 6.4 MB/s
Collecting paramiko~=3.1.0
  Downloading paramiko-3.1.0-py3-none-any.whl (211 kB)
     |████████████████████████████████| 211 kB ...
Collecting pycparser
  Downloading pycparser-3.0-py3-none-any.whl (48 kB)
     |████████████████████████████████| 48 kB 3.2 MB/s
Collecting charset-normalizer<4,>=2
  Downloading charset_normalizer-3.4.7-cp310-cp310-win_amd64.whl (159 kB)
     |████████████████████████████████| 159 kB 6.4 MB/s
Collecting idna<4,>=2.5
  Using cached idna-3.11-py3-none-any.whl (71 kB)
Collecting certifi>=2017.4.17
  Downloading certifi-2026.2.25-py3-none-any.whl (153 kB)
     |████████████████████████████████| 153 kB 6.8 MB/s 
Collecting urllib3<1.27,>=1.21.1
  Downloading urllib3-1.26.20-py2.py3-none-any.whl (144 kB)
     |████████████████████████████████| 144 kB ...
Collecting beautifulsoup4
  Downloading beautifulsoup4-4.14.3-py3-none-any.whl (107 kB)
     |████████████████████████████████| 107 kB 6.8 MB/s
Collecting sniffio
  Using cached sniffio-1.3.1-py3-none-any.whl (10 kB)
Collecting tzlocal!=3.*,>=2.0
  Using cached tzlocal-5.3.1-py3-none-any.whl (18 kB)
Collecting six>=1.4.0
  Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Collecting pytz
  Downloading pytz-2026.1.post1-py2.py3-none-any.whl (510 kB)
     |████████████████████████████████| 510 kB 6.4 MB/s
Collecting colorama
  Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Collecting iniconfig
  Downloading iniconfig-2.3.0-py3-none-any.whl (7.5 kB)
Collecting packaging
  Using cached packaging-26.0-py3-none-any.whl (74 kB)
Collecting tomli>=1.0.0
  Downloading tomli-2.4.1-py3-none-any.whl (14 kB)
Collecting pluggy<2.0,>=0.12
  Downloading pluggy-1.6.0-py3-none-any.whl (20 kB)
Collecting exceptiongroup>=1.0.0rc8
  Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)
Collecting attrs>=19.2.0
  Downloading attrs-26.1.0-py3-none-any.whl (67 kB)
     |████████████████████████████████| 67 kB 2.9 MB/s
Collecting cryptography>=3.3
  Downloading cryptography-46.0.6-cp38-abi3-win_amd64.whl (3.5 MB)
     |████████████████████████████████| 3.5 MB 6.4 MB/s
Collecting pynacl>=1.5
  Downloading pynacl-1.6.2-cp38-abi3-win_amd64.whl (239 kB)
     |████████████████████████████████| 239 kB 6.4 MB/s
Collecting bcrypt>=3.2
  Downloading bcrypt-5.0.0-cp39-abi3-win_amd64.whl (150 kB)
     |████████████████████████████████| 150 kB ...
Collecting cryptography>=3.3
  Downloading cryptography-46.0.5-cp38-abi3-win_amd64.whl (3.5 MB)
     |████████████████████████████████| 3.5 MB ...
  Downloading cryptography-46.0.4-cp38-abi3-win_amd64.whl (3.5 MB)
     |████████████████████████████████| 3.5 MB 6.4 MB/s
  Downloading cryptography-46.0.3-cp38-abi3-win_amd64.whl (3.5 MB)
     |████████████████████████████████| 3.5 MB 6.8 MB/s
  Downloading cryptography-46.0.2-cp38-abi3-win_amd64.whl (3.5 MB)
     |████████████████████████████████| 3.5 MB ...
  Downloading cryptography-46.0.1-cp38-abi3-win_amd64.whl (3.5 MB)
     |████████████████████████████████| 3.5 MB 6.4 MB/s
  Downloading cryptography-46.0.0-cp38-abi3-win_amd64.whl (3.5 MB)
     |████████████████████████████████| 3.5 MB 6.8 MB/s
Collecting typing-extensions>=4.13.2
  Using cached typing_extensions-4.15.0-py3-none-any.whl (44 kB)
Collecting pynacl>=1.5
  Downloading pynacl-1.6.1-cp38-abi3-win_amd64.whl (238 kB)
     |████████████████████████████████| 238 kB 3.3 MB/s
  Downloading pynacl-1.6.0-cp38-abi3-win_amd64.whl (238 kB)
     |████████████████████████████████| 238 kB ...
Collecting tzdata
  Downloading tzdata-2026.1-py2.py3-none-any.whl (348 kB)
     |████████████████████████████████| 348 kB 6.4 MB/s
Collecting soupsieve>=1.6.1
  Downloading soupsieve-2.8.3-py3-none-any.whl (37 kB)
Installing collected packages: pycparser, tzdata, typing-extensions, soupsieve, cffi, urllib3, tzlocal, tomli, sniffio, six, pytz, pynacl, pluggy, packaging, iniconfig, idna, exceptiongroup, cryptography, colorama, charset-normalizer, certifi, beautifulsoup4, bcrypt, attrs, setuptools, requests, pytest, pynng, protocol, protobuf, paramiko, google, APScheduler, aenum
  Attempting uninstall: setuptools
    Found existing installation: setuptools 57.4.0
    Uninstalling setuptools-57.4.0:
      Successfully uninstalled setuptools-57.4.0
Successfully installed APScheduler-3.10.4 aenum-3.1.17 attrs-26.1.0 bcrypt-5.0.0 beautifulsoup4-4.14.3 certifi-2026.2.25 cffi-1.15.1 charset-normalizer-3.4.7 colorama-0.4.6 cryptography-46.0.0 exceptiongroup-1.3.1 google-3.0.0 i
dna-3.11 iniconfig-2.3.0 packaging-26.0 paramiko-3.1.0 pluggy-1.6.0 protobuf-4.23.0 protocol-0.1.0 pycparser-3.0 pynacl-1.6.0 pynng-0.7.4 pytest-7.2.2 pytz-2026.1.post1 requests-2.28.2 setuptools-67.6.1 six-1.17.0 sniffio-1.3.1 soupsieve-2.8.3 tomli-2.4.1 typing-extensions-4.15.0 tzdata-2026.1 tzlocal-5.3.1 urllib3-1.26.20
WARNING: You are using pip version 21.2.3; however, version 26.0.1 is available.
You should consider upgrading via the 'C:\Users\Holo\PycharmProjects\Manipularor + ML\.venv\Scripts\python.exe -m pip install --upgrade pip' command.
```
Теперь установим сам софт для манипулятора, в нем есть все команды и методы для управления.
```
(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>pip install "Agilebot.Robot.SDK.A-1.7.1.3+ca90c030.20250703-py3-none-any.whl"
Processing c:\users\holo\pycharmprojects\manipularor + ml\agilebot.robot.sdk.a-1.7.1.3+ca90c030.20250703-py3-none-any.whl
Installing collected packages: Agilebot.Robot.SDK.A
Successfully installed Agilebot.Robot.SDK.A-1.7.1.3+ca90c030.20250703
WARNING: You are using pip version 21.2.3; however, version 26.0.1 is available.
You should consider upgrading via the 'C:\Users\Holo\PycharmProjects\Manipularor + ML\.venv\Scripts\python.exe -m pip install --upgrade pip' command.
```
Дальше будет череда неудачных запусков скрипта для управления манипулятором. Около 20 минут не получалось понять, в чем вообще проблема. Думал, что проблема в том, что я подключаюсь к манипулятору через вайфай, а не через кабель, но дело было не в этом. И как бывает в создании любого проекта - если что-то да потыкать, то рано или поздно все заработает и да, он начал реагировать на команды. Самое обидное, что я плохо помню, что именно было исправлено. Мне помог мой коллега - Олег, за это ему огромное спасибо. Думаю, стоит у него спросить. Как узнаю точную причину - сразу скажу).
```
(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>python pyDHgripper\Basa.py     
Copyright © 2016 Shanghai Agilebot Robotics Co., Ltd., All rights reserved.
SDK Version: 1.7.1.3.ca90c030
Traceback (most recent call last):
  File "C:\Users\Holo\PycharmProjects\Manipularor + ML\pyDHgripper\Basa.py", line 11, in <module>
    assert ret == StatusCodeEnum.OK
AssertionError

(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>python pyDHgripper\Basa.py
Copyright © 2016 Shanghai Agilebot Robotics Co., Ltd., All rights reserved.
SDK Version: 1.7.1.3.ca90c030
Robot Model: GBT-C5A
The controller version: 1.2.7 is too low, recommended upgrade to 1.2.7.20241027.3d381e02.
警告：控制器版本：1.2.7 太低，建议升级到 1.2.7.20241027.3d381e02。
Controller version: 1.2.7.20241027.3d381e02
Traceback (most recent call last):
  File "C:\Users\Holo\PycharmProjects\Manipularor + ML\pyDHgripper\Basa.py", line 25, in <module>
    motion_pose.cart.Data.position.x = -108 + X_OFFSET
AttributeError: 'MotionPose' object has no attribute 'cart'

(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>python pyDHgripper\Basa.py
Copyright © 2016 Shanghai Agilebot Robotics Co., Ltd., All rights reserved.
SDK Version: 1.7.1.3.ca90c030
Robot Model: GBT-C5A
The controller version: 1.2.7 is too low, recommended upgrade to 1.2.7.20241027.3d381e02.
警告：控制器版本：1.2.7 太低，建议升级到 1.2.7.20241027.3d381e02。
Controller version: 1.2.7.20241027.3d381e02
Traceback (most recent call last):
  File "C:\Users\Holo\PycharmProjects\Manipularor + ML\pyDHgripper\Basa.py", line 33, in <module>
    assert ret == StatusCodeEnum.OK
AssertionError

(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>python pyDHgripper\Basa.py
Copyright © 2016 Shanghai Agilebot Robotics Co., Ltd., All rights reserved.
SDK Version: 1.7.1.3.ca90c030
Robot Model: GBT-C5A
The controller version: 1.2.7 is too low, recommended upgrade to 1.2.7.20241027.3d381e02.
警告：控制器版本：1.2.7 太低，建议升级到 1.2.7.20241027.3d381e02。
Controller version: 1.2.7.20241027.3d381e02
Traceback (most recent call last):
  File "C:\Users\Holo\PycharmProjects\Manipularor + ML\pyDHgripper\Basa.py", line 33, in <module>
    assert ret == StatusCodeEnum.OK, "move_joint error"
AssertionError: move_joint error

(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>python pyDHgripper\Basa.py
Copyright © 2016 Shanghai Agilebot Robotics Co., Ltd., All rights reserved.
SDK Version: 1.7.1.3.ca90c030
Robot Model: GBT-C5A
The controller version: 1.2.7 is too low, recommended upgrade to 1.2.7.20241027.3d381e02.
警告：控制器版本：1.2.7 太低，建议升级到 1.2.7.20241027.3d381e02。
Controller version: 1.2.7.20241027.3d381e02
Traceback (most recent call last):
  File "C:\Users\Holo\PycharmProjects\Manipularor + ML\pyDHgripper\Basa.py", line 33, in <module>
    assert ret == StatusCodeEnum.OK, f"move_joint error: {ret}"
AssertionError: move_joint error: StatusCodeEnum.MC_COMPUTE_IK_FAIL

(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>python pyDHgripper\Basa.py
Copyright © 2016 Shanghai Agilebot Robotics Co., Ltd., All rights reserved.
SDK Version: 1.7.1.3.ca90c030
Robot Model: GBT-C5A
The controller version: 1.2.7 is too low, recommended upgrade to 1.2.7.20241027.3d381e02.
警告：控制器版本：1.2.7 太低，建议升级到 1.2.7.20241027.3d381e02。
Controller version: 1.2.7.20241027.3d381e02
Traceback (most recent call last):
  File "C:\Users\Holo\PycharmProjects\Manipularor + ML\pyDHgripper\Basa.py", line 33, in <module>
    assert ret == StatusCodeEnum.OK, f"move_joint error: {ret}"
AssertionError: move_joint error: StatusCodeEnum.MC_COMPUTE_IK_FAIL

(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>python pyDHgripper\Basa.py
Copyright © 2016 Shanghai Agilebot Robotics Co., Ltd., All rights reserved.
SDK Version: 1.7.1.3.ca90c030
Traceback (most recent call last):
  File "C:\Users\Holo\PycharmProjects\Manipularor + ML\pyDHgripper\Basa.py", line 12, in <module>
    assert ret == StatusCodeEnum.OK
AssertionError

(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>python pyDHgripper\Basa.py
Copyright © 2016 Shanghai Agilebot Robotics Co., Ltd., All rights reserved.
SDK Version: 1.7.1.3.ca90c030
Traceback (most recent call last):
  File "C:\Users\Holo\PycharmProjects\Manipularor + ML\pyDHgripper\Basa.py", line 12, in <module>
    assert ret == StatusCodeEnum.OK, f"connection error: {ret}"
AssertionError: connection error: StatusCodeEnum.CONTROLLER_ERROR

(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>python pyDHgripper\Basa.py
Copyright © 2016 Shanghai Agilebot Robotics Co., Ltd., All rights reserved.
SDK Version: 1.7.1.3.ca90c030
Robot Model: GBT-C5A
The controller version: 1.2.7 is too low, recommended upgrade to 1.2.7.20241027.3d381e02.
警告：控制器版本：1.2.7 太低，建议升级到 1.2.7.20241027.3d381e02。
Controller version: 1.2.7.20241027.3d381e02
Traceback (most recent call last):
  File "C:\Users\Holo\PycharmProjects\Manipularor + ML\pyDHgripper\Basa.py", line 35, in <module>
    assert ret == StatusCodeEnum.OK, f"move_joint error: {ret}"
AssertionError: move_joint error: StatusCodeEnum.MC_COMPUTE_IK_FAIL

(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>python pyDHgripper\Basa.py                                                    
Copyright © 2016 Shanghai Agilebot Robotics Co., Ltd., All rights reserved.
SDK Version: 1.7.1.3.ca90c030
Robot Model: GBT-C5A
The controller version: 1.2.7 is too low, recommended upgrade to 1.2.7.20241027.3d381e02.
警告：控制器版本：1.2.7 太低，建议升级到 1.2.7.20241027.3d381e02。
Controller version: 1.2.7.20241027.3d381e02
Traceback (most recent call last):
  File "C:\Users\Holo\PycharmProjects\Manipularor + ML\pyDHgripper\Basa.py", line 35, in <module>
    assert ret == StatusCodeEnum.OK, f"move_joint error: {ret}"
AssertionError: move_joint error: StatusCodeEnum.MC_COMPUTE_IK_FAIL

(.venv) C:\Users\Holo\PycharmProjects\Manipularor + ML>python pyDHgripper\Basa.py
Copyright © 2016 Shanghai Agilebot Robotics Co., Ltd., All rights reserved.
SDK Version: 1.7.1.3.ca90c030
Robot Model: GBT-C5A
The controller version: 1.2.7 is too low, recommended upgrade to 1.2.7.20241027.3d381e02.
警告：控制器版本：1.2.7 太低，建议升级到 1.2.7.20241027.3d381e02。
Controller version: 1.2.7.20241027.3d381e02
Traceback (most recent call last):
  File "C:\Users\Holo\PycharmProjects\Manipularor + ML\pyDHgripper\Basa.py", line 35, in <module>
    assert ret == StatusCodeEnum.OK, f"move_joint error: {ret}"
AssertionError: move_joint error: StatusCodeEnum.MC_COMPUTE_IK_FAIL
```
Как вы могли понять, с этого момента он начал исправно работать. После этого мы начали тестировать уже написанный кем-то код для захвата и он тоже исправно работал, что тоже не может не радовать. Также, я пощупал, что из себя представляет обратная кинематика и это очень классная вещь. В коде можно задать координаты и переместить манипулятор в любое место. При запуске скрипта, он переместиться в указанную точку сам - мне не нужно контролировать движение каждой его части.

## Что будем делать дальше ?
1. Оставлю пока что две команды для модели - это движение влево и вправо;
2. Расширю датасет, чтобы модель работала точнее. При необходимости попробую другой метод обучения;
3. Соединю все компоненты воедино: модель + манипулятор + захват.
> Идею с компьютерным зрением пока отложу, не хочу распыляться. Когда все будет работать исправно, тогда можно апгрейднуть систему.

20.04.2026
я сбалансировал классы и сделал их одинакового размера. В коде убрал 
```
Загружено: left=107, right=107, forward=109, backward=112, other=252
Всего примеров: 687
Обучающая выборка: 549, Тестовая выборка: 138
Размерность векторов: 883
Количество итераций обучения: 23

Accuracy: 0.8551
              precision    recall  f1-score   support

       other       0.90      0.84      0.87        51
        left       0.95      0.95      0.95        21
       right       0.95      0.95      0.95        21
     forward       0.69      0.91      0.78        22
    backward       0.79      0.65      0.71        23

    accuracy                           0.86       138
   macro avg       0.86      0.86      0.85       138
weighted avg       0.86      0.86      0.85       138

Confusion matrix:
[[43  1  1  4  2]
 [ 0 20  0  0  1]
 [ 1  0 20  0  0]
 [ 1  0  0 20  1]
 [ 3  0  0  5 15]]
Модели сохранены. Время выполнения: 0.09 сек
```

```
Загружено всего примеров: 400
Примеров на класс: 80
Обучающая выборка: 320, Тестовая: 80

Accuracy: 0.8750
              precision    recall  f1-score   support

       other       0.93      0.88      0.90        16
        left       0.94      0.94      0.94        16
       right       0.74      0.88      0.80        16
     forward       0.82      0.88      0.85        16
    backward       1.00      0.81      0.90        16

    accuracy                           0.88        80
   macro avg       0.89      0.88      0.88        80
weighted avg       0.89      0.88      0.88        80

Confusion matrix:
[[14  0  0  2  0]
 [ 0 15  1  0  0]
 [ 1  1 14  0  0]
 [ 0  0  2 14  0]
 [ 0  0  2  1 13]]
```
модель стала значительно сильнее путать класс. надо как-то исправлять это. Оказывается, я в датасете по невнимательности повставлял такие слова, как: верхний, нижный итд. Надо ж было так проглядеть. 
Несмотря на то, что я почистил датасет от мусорных фраз, матрица ошибок никак не изменилась, а знаете почему? - я все это время запускал старый скрипт, параллельно занимаясь рефакторингом другого..
```
Загружено: forward=80, backward=80, left=80, right=80, other=80
Всего примеров: 400
Обучающая выборка: 320,
тестовая: 80

Accuracy: 0.9250 (92.50%)
Log-loss: 0.7845
Cohen's Kappa: 0.9062
              precision    recall  f1-score   support

       other       1.00      0.88      0.93        16
     forward       0.89      1.00      0.94        16
    backward       0.88      0.88      0.88        16
        left       0.94      0.94      0.94        16
       right       0.94      0.94      0.94        16

    accuracy                           0.93        80
   macro avg       0.93      0.93      0.92        80
weighted avg       0.93      0.93      0.92        80


Confusion matrix:
[[14  0  2  0  0]
 [ 0 16  0  0  0]
 [ 0  1 14  1  0]
 [ 0  0  0 15  1]
 [ 0  1  0  0 15]]
```

```
> перемести кубик
❌ Действие не распознано | Уверенность: 24.18%
> перемести кубик вперед
✅ Действие выполнено: переместить объект влево | Уверенность: 75.88%
> перетащи объект влево 
❌ Действие не распознано | Уверенность: 36.87%
```

```
Загружено: forward=80, backward=80, left=80, right=80, other=80
Всего примеров: 400
Обучающая выборка: 320,
тестовая: 80

Accuracy: 0.9250 (92.50%)
Log-loss: 0.8176
Cohen's Kappa: 0.9062
              precision    recall  f1-score   support

       other       1.00      0.88      0.93        16
     forward       0.89      1.00      0.94        16
    backward       1.00      0.88      0.93        16
        left       0.94      0.94      0.94        16
       right       0.83      0.94      0.88        16

    accuracy                           0.93        80
   macro avg       0.93      0.93      0.93        80
weighted avg       0.93      0.93      0.93        80


Confusion matrix:
[[14  0  0  0  2]
 [ 0 16  0  0  0]
 [ 0  1 14  1  0]
 [ 0  0  0 15  1]
 [ 0  1  0  0 15]]

ROC-AUC: 0.9898
```
