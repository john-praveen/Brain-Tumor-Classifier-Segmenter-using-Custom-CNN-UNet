# 🧠 Brain Tumor Detection, Classification & Segmentation

A complete deep-learning pipeline for analyzing brain MRI images —
binary tumor detection, tumor-type classification (Meningioma, Glioma, Pituitary), and pixel-wise semantic segmentation using a custom UNet. Includes a Streamlit app so anyone can try the model from the browser.

---

## 🚀 Quick start

```bash
pip install -r requirements.txt
cd streamlit_app
streamlit run app.py
```

Open your browser at the address Streamlit prints (usually [http://localhost:8501](http://localhost:8501)).

---

## ✅ Features

* **Binary Tumor Classification** — Tumor vs No Tumor.
* **Tumor Type Identification** — Meningioma, Glioma, Pituitary.
* **Semantic Segmentation** — Custom-built U-Net architecture that outputs tumor masks.
* **Streamlit App** — Upload an MRI and get predictions + segmentation in the browser.
* Clean, modular project structure so you can re-train, evaluate, or extend easily.

---

## 📁 Repository structure

```
├─ streamlit_app/        # Streamlit web app (app.py)
├─ src/                  # Model code: UNet, classifiers, training + utils
├─ models/               # Trained model checkpoints (not in repo unless added)
├─ notebooks/            # Jupyter notebooks for experiments and EDA
├─ sample/               # Sample MRI images you can try out
├─ requirements.txt
└─ README.md
```

> The `src/` folder contains the neural network code you built for:
>
> * custom **UNet** segmentation model, and
> * tumor classifiers (binary + 3-class tumor type classifier).

---

## 🧩 Model details

* **Segmentation network:** Custom U‑Net (encoder–decoder with skip connections). Implemented in **TensorFlow / Keras**.
* **Classification networks:** Lightweight CNN heads trained for binary tumor detection and 3-way tumor type classification.
* **Training dataset:** Models were trained on a Kaggle dataset (link below). Training was performed on Kaggle (or your preferred GPU environment).

**Kaggle dataset:**
`<INSERT_KAGGLE_DATASET_URL_HERE>`

(Replace the placeholder with the Kaggle dataset link you want to credit.)

---

## 🖼️ Demo images (how to add)

Add sample images into a repository folder `images/` (create it at the repo root). This README includes two visual slots — replace the example paths with your actual images.

* **Input example** — *what the user will upload*: `images/input_sample.png`

```
![Input MRI sample](images/input_sample.png)
```

* **Predicted output** — *segmentation mask overlay + predicted tumor category*: `images/predicted_output.png`

```
![Predicted segmentation and classification](images/predicted_output.png)
```

If you want a nicer layout, create a tiny `assets/` folder and reference the images from there. GitHub will render them automatically in the README.

---

## 🧪 How to use

1. Run the Streamlit app (see Quick start).
2. In the web UI, click **Browse files** and select an MRI (or try a sample from `sample/`).
3. Click **Predict**. The app will display:

   * Tumor present? (Yes / No)
   * If present: tumor type (Meningioma / Glioma / Pituitary)
   * Segmentation mask overlaid on the original MRI

---

## 🛠️ Reproduce training

To retrain the models locally or on Kaggle:

1. Prepare the dataset with the same folder layout your `src/data` loader expects (see `src/utils/data_loader.py`).
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the training script (example):

```bash
python src/train_segmentation.py --config configs/unet_config.yaml
python src/train_classifier.py --config configs/classifier_config.yaml
```

Training hyperparameters, augmentation, and evaluation metrics are controlled by the YAML config files in `configs/`.

---

## 🧾 Notes / Tips

* I used **TensorFlow** and trained the models on **Kaggle** (GPU runtime). If you want Docker or Colab notebooks to reproduce the environment, tell me and I can add them.
* For segmentation visualisation I overlay the predicted mask using simple alpha blending — see `src/visualize/overlay.py`.
* If you plan to publish the model, include model cards and details about the dataset splits (train/val/test), performance metrics (Dice, IoU, accuracy), and known limitations.

---

## 📌 Example metrics (add yours)

Replace these with your actual numbers after evaluation:

* Segmentation — Dice score: `0.86`
* Classification (binary) — Accuracy: `0.95`
* Classification (3-way) — Accuracy: `0.89`

---

## 🙋‍♂️ Author

Praveen V.V.J — 2025

If you'd like, I can:

* add the actual Kaggle link into this README,
* replace placeholders with your real images (you can upload them or give paths),
* generate a polished `README.png` preview layout, or
* add license and citation text.

---

## 🧾 License

*Add your preferred license (MIT, Apache-2.0, etc.) here.*

---

## 📬 Contact

If you want me to fill in the Kaggle URL and the two output images here, paste them and I will update th
