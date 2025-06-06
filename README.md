# simple_detectron2

**simple_detectron2** — минималистичный шаблон для обучения и применения моделей детекции объектов на базе [Detectron2](https://github.com/facebookresearch/detectron2), протестирован для задач детекции и классификации наночастиц.

---

## 🚀 Возможности

- Быстрый старт для обучения моделей на своих данных
- Поддержка кастомных датасетов (COCO)
- Скрипты для обучения, инференса и визуализации
- Лёгкая настройка конфигураций
- Минималистичный интерфейс для работы с моделями

---

## 📦 Установка

1. **Клонируйте репозиторий:**

    ```
    git clone https://github.com/arturgromenkov/simple_detectron2.git
    cd simple_detectron2
    ```

2. **Установите зависимости:**

    ```
    pip install -r requirements.txt
    ```

3. **Установите Detectron2:**

    Ознакомьтесь с [официальной инструкцией](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) или выполните:

    ```
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```

---

## 🗂️ Структура проекта
```
📦nanodetect
 ┣ 📂example_files
 ┃ ┣ 📂53000X_train
 ┃ ┃ ┣ 📜53000X00.png
 ┃ ┃ ┣ 📜...png
 ┃ ┃ ┣ 📜53000X998.png
 ┃ ┃ ┣ 📜53000X999.png
 ┃ ┃ ┣ 📜data.json
 ┃ ┃ ┣ 📜train.json
 ┃ ┃ ┗ 📜val.json
 ┃ ┗ 📜callback.py
 ┣ 📂scripts
 ┃ ┣ 📜config.py
 ┃ ┣ 📜export_model.py
 ┃ ┣ 📜test_exported_model.py
 ┃ ┣ 📜test_model.py
 ┃ ┗ 📜train_model.py
 ┣ 📜app.py
 ┗ 📜app_config.py
```

---

## ⚡ Быстрый старт

### 1. Подготовьте ваш датасет в формате COCO, пример датасета есть в папке example_files
### 2. Отредактируйте файл config.py под ваши нужды
```
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.INPUT.MIN_SIZE_TRAIN = (416,)
    cfg.INPUT.MAX_SIZE_TRAIN = 416
    cfg.INPUT.MIN_SIZE_TEST = 416
    cfg.INPUT.MAX_SIZE_TEST = 416
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = ("dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 20_000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # MUST BE EQUAL TO CLASSES
    cfg.OUTPUT_DIR = output_folder
    cfg.TEST.EVAL_PERIOD = 1_500
    cfg.SOLVER.CHECKPOINT_PERIOD = 1_500

    cfg.MODEL.DEVICE = device
```
### 3. Запустите скрипт обучения
```
python scripts/train_model.py path/to/train.json path/to/val.json path/to/images_folder path/to/output_folder
```
### 4. Наблюдайте за процессом обучения
```
tensorboard --logdir path/to/output_folder
```
### 5. После окончания тренировки проверьте работоспособность вашей модели (только для задач детекций)
```
python scripts/test_model.py path/to/model_checkpoint.pth path/to/image.jpg
```
### 6. Экспортируйте модель в формат torchscript
```
python scripts/export_model.py path/to/model_checkpoint.pth path/to/exported_model.pt cuda
```
### 7. Чтобы использовать простой интерфейс для экспортированной модели, отредактируйте файл app_config.py
```
# App name
app_name = 'my_model'
# Absolute path to your model
model_path = '..\model.pt'
# Absolute path to your callback.py
call_back_path = '..callback.py'
# Model device
device='cuda'
...
```
### 8. Запустите интерфейс
```
python app.py
```
![image](https://github.com/user-attachments/assets/9092b4c6-1ceb-44d5-819b-d5bc39bb7c92)
![image](https://github.com/user-attachments/assets/8a521b2b-aa85-4fbd-8be1-16cfafc5fdd6)
![image](https://github.com/user-attachments/assets/859f3b31-c7a7-4a22-beb4-52fa4dab5bfc)

## ⚙️ Требования

- Python 3.8+
- PyTorch 1.10+
- Detectron2
- Другие зависимости из `requirements.txt`

---

## 📚 Полезные ссылки

- [Документация Detectron2](https://detectron2.readthedocs.io/)
- [Официальный репозиторий Detectron2](https://github.com/facebookresearch/detectron2)
