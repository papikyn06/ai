import os
from typing import Dict, Any, List, Optional

import numpy as np
import tensorflow as tf
from PIL import Image

TRAFFIC_SIGNS_RU = {
    0: "Ограничение скорости 20 км/ч",
    1: "Ограничение скорости 30 км/ч",
    2: "Ограничение скорости 50 км/ч",
    3: "Ограничение скорости 60 км/ч",
    4: "Ограничение скорости 70 км/ч",
    5: "Ограничение скорости 80 км/ч",
    6: "Конец ограничения скорости 80 км/ч",
    7: "Ограничение скорости 100 км/ч",
    8: "Ограничение скорости 120 км/ч",
    9: "Запрет обгона",
    10: "Запрет обгона для грузовиков",
    11: "Приоритет на перекрёстке",
    12: "Главная дорога",
    13: "Уступить дорогу",
    14: "Стоп",
    15: "Движение запрещено",
    16: "Запрет для грузовиков",
    17: "Въезд запрещён",
    18: "Опасное место",
    19: "Опасный поворот налево",
    20: "Опасный поворот направо",
    21: "Двойной поворот",
    22: "Неровная дорога",
    23: "Скользкая дорога",
    24: "Сужение дороги",
    25: "Дорожные работы",
    26: "Светофор",
    27: "Пешеходы",
    28: "Дети",
    29: "Велосипедный транспорт",
    30: "Скользкая дорога",
    31: "Дикие животные",
    32: "Конец ограничений",
    33: "Движение направо",
    34: "Движение налево",
    35: "Движение прямо",
    36: "Движение прямо и направо",
    37: "Движение прямо и налево",
    38: "Объезд справа",
    39: "Объезд слева",
    40: "Круговое движение",
    41: "Конец запрета обгона",
    42: "Конец запрета обгона грузовиков",
}

MODEL_CANDIDATES = [
    "best_traffic_signs_model.keras",
    "traffic_signs_model.keras",
]


def load_traffic_model() -> tf.keras.Model:
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            model = tf.keras.models.load_model(p)
            print(f"✓ Загружена модель: {p}")
            return model
    raise FileNotFoundError(
        "Не найден файл модели. Ожидалось: " + ", ".join(MODEL_CANDIDATES)
    )


model = load_traffic_model()


def preprocess_image(image_path: str, target_size: int = 48) -> np.ndarray:
    """
    - RGB
    - resize до 48x48
    - float32
    - нормализация /255
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _softmax_mean(preds: np.ndarray) -> np.ndarray:
    # preds shape: (N, 43)
    return preds.mean(axis=0)


def predict_traffic_sign(
        image_path: str,
        target_size: int = 48,
        tta: bool = True,
        top_k: int = 5,
        confidence_threshold: float = 0.70,
) -> Optional[Dict[str, Any]]:
    """
    - class_id
    - name_ru
    - confidence
    - top_k (список)
    - is_confident (по порогу)
    """
    try:
        x = preprocess_image(image_path, target_size=target_size)

        if tta:

            variants = [
                x,
                np.clip(x * 1.05, 0, 1),
                np.clip(x * 0.95, 0, 1),
            ]
            batch = np.stack(variants, axis=0)
            preds = model.predict(batch, verbose=0)
            probs = _softmax_mean(preds)
        else:
            batch = np.expand_dims(x, axis=0)
            probs = model.predict(batch, verbose=0)[0]

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        # top-k
        k = int(max(1, min(top_k, probs.shape[0])))
        top_idx = np.argsort(probs)[-k:][::-1]
        top = []
        for i in top_idx:
            top.append({
                "class": int(i),
                "name_ru": TRAFFIC_SIGNS_RU.get(int(i), "Неизвестный знак"),
                "confidence": float(probs[i]),
            })
        print(class_id)
        return {
            "class": class_id,
            "name_ru": TRAFFIC_SIGNS_RU.get(class_id, "Неизвестный знак"),
            "confidence": confidence,
            "is_confident": confidence >= confidence_threshold,
            "top": top,
        }

    except Exception as e:
        print(f"ОШИБКА: {e}")
        return None


# ----------------------------
# BATCH
# ----------------------------
def predict_batch(
        image_paths: List[str],
        **kwargs
) -> List[Dict[str, Any]]:
    out = []
    for p in image_paths:
        r = predict_traffic_sign(p, **kwargs)
        if r is not None:
            out.append({"path": p, "prediction": r})
    return out


if __name__ == "__main__":
    test_image = "images/main_road.jpg"

    if not os.path.exists(test_image):
        print(f"Файл '{test_image}' не найден. Укажите корректный путь.")
    else:
        res = predict_traffic_sign(
            test_image,
            tta=True,
            top_k=5,
            confidence_threshold=0.70
        )
        print(res)
