# Iris ONNX Android Classifier

An Android application demonstrating real-time Machine Learning inference using **ONNX Runtime for Android** in Java.

The app uses a scikit-learn **Random Forest model** (`rf_iris.onnx`) exported via `skl2onnx` to classify Iris flower species based on physical measurements.

---

## 🌟 Features

* **Interactive Material UI**: Sliders for instant adjustment of feature values:
  * **Sepal Length** (4.0 cm – 8.0 cm)
  * **Sepal Width** (2.0 cm – 4.5 cm)
  * **Petal Length** (1.0 cm – 7.0 cm)
  * **Petal Width** (0.1 cm – 2.5 cm)
* **Real-time Inference**: Runs model predictions asynchronously on a background thread as slider values change or on button click.
* **Probability Visualizations**: Displays predicted species (*Setosa*, *Versicolor*, or *Virginica*) with percentage progress bars.
* **Android 15+ Ready**: Configured with `useLegacyPackaging` for 16 KB page-size memory compatibility across devices and emulators.

---

## 🛠️ Tech Stack & Model Information

* **Language**: Java
* **UI Framework**: AndroidX & Google Material Components (`com.google.android.material`)
* **ML Library**: [ONNX Runtime Android](https://onnxruntime.ai/) (`com.microsoft.onnxruntime:onnxruntime-android`)

### Model Signature (`rf_iris.onnx`)

| Name | Type | Description |
| :--- | :--- | :--- |
| **Input (`X`)** | `float32 [N, 4]` | `[sepal_length, sepal_width, petal_length, petal_width]` |
| **Output (`output_label`)** | `int64 [N]` | Predicted class index (0: setosa, 1: versicolor, 2: virginica) |
| **Output (`output_probability`)** | `seq(map(int64, float))` | ZipMap sequence of class probabilities |

---

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Wallis16/onnx-java.git
   ```
2. **Open in Android Studio**:
   Open Android Studio and choose **File > Open** to select the repository folder.
3. **Build & Run**:
   Sync Gradle and deploy to an Android device or emulator running API 24 or higher.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
