from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import torch.nn.functional as F
import math
import logging

# Настройка логов
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Обработчик для корневого пути
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI server!"}

# Путь к папке с эталонными изображениями
DATASET_PATH = "C:/Users/Вилена/PycharmProjects/VKR/dataset_copy"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Путь {DATASET_PATH} не найден!")

# Определение архитектуры сиамской сети
class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128 * 37 * 37, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

# Загрузка модели
MODEL_PATH = "siamese_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Преобразование изображения
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

# Загрузка эталонных изображений
reference_images = {}

if not os.path.exists(DATASET_PATH):
    raise ValueError(f"Путь {DATASET_PATH} не существует!")

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feature_vector = model.forward_once(img_tensor)
                reference_images[img_name] = (feature_vector, folder)
            except PermissionError as e:
                logger.error(f"Ошибка доступа к файлу {img_path}: {e}")
            except Exception as e:
                logger.error(f"Ошибка загрузки {img_path}: {e}")

if not reference_images:
    raise ValueError("Эталонные изображения не загружены!")

logger.debug(f"Количество эталонных изображений: {len(reference_images)}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        logger.debug("Начало обработки запроса")

        # Загружаем изображение пользователя
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Вычисляем вектор признаков загруженного изображения
        with torch.no_grad():
            query_vector = model.forward_once(image)

        # Поиск наиболее похожего изображения в эталонных данных
        best_match = None
        min_distance = float("inf")

        for img_name, (ref_vector, folder) in reference_images.items():
            distance = F.pairwise_distance(query_vector, ref_vector).item()
            if distance < min_distance:
                min_distance = distance
                best_match = folder

        # Проверка на NaN и Infinity
        if math.isnan(min_distance) or math.isinf(min_distance):
            logger.warning(f"Недопустимое значение distance: {min_distance}")
            min_distance = float("inf")  # или другое значение по умолчанию

        logger.debug(f"Результат: {best_match}, Расстояние: {min_distance}")

        return {
            "recognized_object": best_match,
            "distance": min_distance
        }
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)