###Отчет
+ Первоначально я посмотрел на изображения в датасетах и увидел в них помехи. Решил преобработать изображения следующим образом: для каждой клетки с насыщенным на 100% компонентном цвета я этот компонент пересчитывал как среднее компонентов соседних клеток. В результате получились более плавные изображения
+ Далее я посмотрел, какие в pytorch есть доступные архитектураы моделей, и выбрал resnet18. Первоначально обучал модель на SGD. Потом поподбирал другие оптимизаторы (вроде Adam и QHAdam) и остановился на QHAdam. Для каждого оптимизатора я подбирал learning rate, перебирая его и сравнивая точность после 5 эпох обучения. resnet18, обученный с помощью QHAdam, набрал около 87-88% на тестировании.
+ Дальнейший подбор параметров мне не помогал, поэтому я стал искать другие возможный варианты моделей. В частности я попробовал resnet34 и resnet50. Но так как это более глубокие модели, обучались они сильно медленно. Я решил, что такие глубокие модели мне не нужны. 
+ Тогда я стал искать сравнительно небольшие модели и нашел статью https://paperswithcode.com/paper/an-ensemble-of-simple-convolutional-neural, в которой были описаны модели M3, M5, M7. Лучший результат давала M5, поэтому я выбрал ее и дотренировал до 90% на валидации. Решение, полученное с помощью M5, получило около 90% на валидации.
+ Далее я понял, что модель хорошо обучается для train_dataset (больше чем на 99%), и плохо для val_dataset. Поэтому я решил каждое изображение менять каким-то образом. Для начала я использовал augmenter из pytorch, это давало 91% на валидации. После я вместо augmenter стал использовать color_jitter(torch.transforms, меняет цветовую окраску), cat_out(самописная функция, рандомно выбирающая каой-нибудь небольшой квадрат, на котором значения усредняются) и random_perspective(torch.transforms, рандомно поворачивает изображение). Теперь модель набирала 91% на валидации.
+ Следущее, что я решил реализовать - это ансамбль нескольких моделей. В нем каждой модели мы сопоставляем какой-то вес, суммируем ответы моделей("вероятности" для каждого кластера), помноженные на веса, и берем кластер с максимальной суммой. Изначально все веса я установил равными. Такой ансамбль набирал 92% на валидации и 91.1% на тестировании.
+ Далее я решил снова искать архитектуры моделей. Так я нашел CnnFnnModel, которая после подбора параметров оптимизатора, давала 93.4% на валидации и 92% на тестировании. 
+ На данном шаге встала проблема недообучения (на train_dataset точность не дохоила до 99%). Тогда я стал изменять архитектуры моделей (меняя функции активации и изменяя количество слоев). Изменение функций активации мне ничего не дало. Однако увеличение количества слоев на M5 (увеличенная модель - M4) увеличило точность на валидации до 93%. 
+ На данном шаге у меня накопилось достаточное количество моделей с точностью 92-93% и я решил снова составить ансамбль (теперь сначала брал ансамбль с одинаковыми весами, а потом несколько раз пробовал брать рандомные веса, и брал наиболее подходящие веса). Он набирал 92.9% на тестировании.
+ Я попробовал подбирать коэффициенты ансамбля обучением. Но каждый раз все сводилось к тому, что у модели, дающий лучший результат, коэффициент был близкий к 1, а у остальный - близкий к 0.
+ Далее я попробовал различные модели из pytorch(вроде googleNet, denseNet, mobileNet). Но данные модели были большими и тренировались долго.
+ Поэтому я решил увеличить количество слоев CnnFnnModel. Увеличенная модель(CnnFnnModel_deeper) стала набирать 94% на валидации.
+ Далее я захотел более оптимально подбирать параметры обучения и мне посоветовали библиотеку optuna. С помощью нее мне удалось увеличить точность моделей на валидации: M5 - 93.7%, M4 - 94.2%, CnnFnnModel_deeper - 95% на валидации
+ И наконец, я снова собрал ансамбль. Он набрал 95.2% на валидации и 93.9% на тестировании. 
+ Незадолго до конца соервнования я захотел попробовать с помощью optune подбирать веса для моделей в ансамбле. Но реализовал я это уже по завершении соревнования, да и лучшего результата мне это не дало.
