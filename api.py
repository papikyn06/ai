import base64
import io
import os
from typing import Dict, Any, List, Optional

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
"""
ТОП-10 классов с наибольшей ошибкой:
Class 24: acc= 97.50% | err=  2.50% | samples=40
Class  4: acc= 99.66% | err=  0.34% | samples=297
Class 38: acc= 99.68% | err=  0.32% | samples=311
Class  2: acc= 99.70% | err=  0.30% | samples=338
Class 42: acc=100.00% | err=  0.00% | samples=36
Class 11: acc=100.00% | err=  0.00% | samples=198
Class 17: acc=100.00% | err=  0.00% | samples=167
Class 16: acc=100.00% | err=  0.00% | samples=63
Class 15: acc=100.00% | err=  0.00% | samples=95
Class 14: acc=100.00% | err=  0.00% | samples=117
"""
# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = 48
TOP_K_DEFAULT = 5
CONF_THRESHOLD_DEFAULT = 0.70

MODEL_CANDIDATES = [
    "best_traffic_signs_model.keras",
    "traffic_signs_model.keras",
]


def load_traffic_model() -> Optional[tf.keras.Model]:
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            try:
                m = tf.keras.models.load_model(p)
                print(f"✓ Модель загружена: {p}")
                return m
            except Exception as e:
                print(f"✗ Ошибка загрузки {p}: {e}")
    print("✗ Модель не найдена. Ожидалось: " + ", ".join(MODEL_CANDIDATES))
    return None


model = load_traffic_model()


def _read_image_from_request() -> Image.Image:
    """
    1) multipart/form-data: file=<image>
    2) JSON: {"image": "<base64>" } (
    """
    if "file" in request.files:
        file = request.files["file"]
        return Image.open(file.stream)

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        if "image" in payload:
            img_b64 = payload["image"]
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            raw = base64.b64decode(img_b64)
            return Image.open(io.BytesIO(raw))

    raise ValueError("No image provided. Use multipart field 'file' or JSON field 'image' (base64).")


def preprocess_image_pil(img: Image.Image, target_size: int = 48) -> np.ndarray:
    """
    - RGB
    - resize до 48x48
    - /255
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def predict_with_tta(x: np.ndarray) -> np.ndarray:
    variants = [
        x,
        np.clip(x * 1.05, 0, 1),
        np.clip(x * 0.95, 0, 1),
    ]
    batch = np.stack(variants, axis=0)
    preds = model.predict(batch, verbose=0)  # (N, 43)
    probs = preds.mean(axis=0)
    return probs


def build_top_k(probs: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    k = int(max(1, min(top_k, probs.shape[0])))
    idxs = np.argsort(probs)[-k:][::-1]
    out = []
    for i in idxs:
        out.append({
            "class": int(i),
            "name_ru": TRAFFIC_SIGNS_RU.get(int(i), "Неизвестный знак"),
            "confidence": float(probs[i]),
        })
    print(int(i), float(probs[i]))
    return out


@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "service": "Traffic Sign Recognition API",
        "model_loaded": model is not None,
        "model_files_checked": MODEL_CANDIDATES,
        "endpoints": {
            "GET /health": "health check",
            "POST /predict": "predict one image (multipart 'file' or JSON base64 'image')",
            "POST /batch-predict": "predict multiple images (multipart 'files')",
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if model is not None else "error",
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    try:
        # параметры
        top_k = TOP_K_DEFAULT
        conf_thr = CONF_THRESHOLD_DEFAULT
        tta = True

        if request.is_json:
            payload = request.get_json(silent=True) or {}
            top_k = int(payload.get("top_k", top_k))
            conf_thr = float(payload.get("confidence_threshold", conf_thr))
            tta = bool(payload.get("tta", tta))
        else:
            # multipart/form-data
            top_k = int(request.form.get("top_k", top_k))
            conf_thr = float(request.form.get("confidence_threshold", conf_thr))
            tta = request.form.get("tta", "true").lower() in ("1", "true", "yes", "y")

        img = _read_image_from_request()
        x = preprocess_image_pil(img, target_size=IMG_SIZE)

        if tta:
            probs = predict_with_tta(x)
        else:
            probs = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        return jsonify({
            "success": True,
            "prediction": {
                "class": class_id,
                "name_ru": TRAFFIC_SIGNS_RU.get(class_id, "Неизвестный знак"),
                "confidence": confidence,
                "is_confident": confidence >= conf_thr
            },
            "top_predictions": build_top_k(probs, top_k=top_k),
            "meta": {
                "tta": tta,
                "top_k": top_k,
                "confidence_threshold": conf_thr,
                "img_size": IMG_SIZE
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    try:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"success": False, "error": "No files provided. Use multipart field 'files'."}), 400

        top_k = int(request.form.get("top_k", TOP_K_DEFAULT))
        conf_thr = float(request.form.get("confidence_threshold", CONF_THRESHOLD_DEFAULT))
        tta = request.form.get("tta", "true").lower() in ("1", "true", "yes", "y")

        results = []
        for f in files:
            try:
                img = Image.open(f.stream)
                x = preprocess_image_pil(img, target_size=IMG_SIZE)

                if tta:
                    probs = predict_with_tta(x)
                else:
                    probs = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]

                class_id = int(np.argmax(probs))
                confidence = float(probs[class_id])

                results.append({
                    "filename": f.filename,
                    "success": True,
                    "prediction": {
                        "class": class_id,
                        "name_ru": TRAFFIC_SIGNS_RU.get(class_id, "Неизвестный знак"),
                        "confidence": confidence,
                        "is_confident": confidence >= conf_thr
                    },
                    "top_predictions": build_top_k(probs, top_k=top_k),
                })

            except Exception as e:
                results.append({
                    "filename": f.filename,
                    "success": False,
                    "error": str(e)
                })

        return jsonify({
            "success": True,
            "total": len(files),
            "results": results,
            "meta": {
                "tta": tta,
                "top_k": top_k,
                "confidence_threshold": conf_thr,
                "img_size": IMG_SIZE
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ЗАПУСК API")
    print("=" * 70)
    print("http://localhost:5000")
    print("GET  /health")
    print("POST /predict")
    print("POST /batch-predict")
    print("=" * 70 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
