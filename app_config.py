import importlib.util
import sys
import os


# App name
app_name = 'my_model'
# Absolute path to your model
model_path = 'F:\WindowsDiplomaProject_NON_REJECT\model.pt'
# Absolute path to your callback.py
call_back_path = 'F:\WindowsDiplomaProject_NON_REJECT\\nanodetect\example_files\\callback.py'
# Model device
device='cuda'
# Do not change below if your function name is still callback
function_name = "callback"

def import_function_from_path(file_path, function_name):
    # Получаем имя модуля из файла
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Создаем спецификацию модуля
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Не удалось создать спецификацию для {file_path}")
    
    # Создаем модуль из спецификации
    module = importlib.util.module_from_spec(spec)
    
    # Загружаем модуль
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Получаем функцию по имени
    func = getattr(module, function_name)
    return func

callback_func = import_function_from_path(call_back_path, function_name)
