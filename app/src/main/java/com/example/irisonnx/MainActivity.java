package com.example.irisonnx;

import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.progressindicator.LinearProgressIndicator;
import com.google.android.material.slider.Slider;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "IrisOnnx";

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private IrisClassifier classifier;

    private Slider sliderSepalLength;
    private Slider sliderSepalWidth;
    private Slider sliderPetalLength;
    private Slider sliderPetalWidth;

    private TextView tvSepalLengthValue;
    private TextView tvSepalWidthValue;
    private TextView tvPetalLengthValue;
    private TextView tvPetalWidthValue;

    private Button btnPredict;

    private TextView tvPredictedClass;
    private TextView tvSetosaProb;
    private TextView tvVersicolorProb;
    private TextView tvVirginicaProb;

    private LinearProgressIndicator progressSetosa;
    private LinearProgressIndicator progressVersicolor;
    private LinearProgressIndicator progressVirginica;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initViews();
        initClassifier();
        setupListeners();
    }

    private void initViews() {
        sliderSepalLength = findViewById(R.id.sliderSepalLength);
        sliderSepalWidth = findViewById(R.id.sliderSepalWidth);
        sliderPetalLength = findViewById(R.id.sliderPetalLength);
        sliderPetalWidth = findViewById(R.id.sliderPetalWidth);

        tvSepalLengthValue = findViewById(R.id.tvSepalLengthValue);
        tvSepalWidthValue = findViewById(R.id.tvSepalWidthValue);
        tvPetalLengthValue = findViewById(R.id.tvPetalLengthValue);
        tvPetalWidthValue = findViewById(R.id.tvPetalWidthValue);

        btnPredict = findViewById(R.id.btnPredict);

        tvPredictedClass = findViewById(R.id.tvPredictedClass);
        tvSetosaProb = findViewById(R.id.tvSetosaProb);
        tvVersicolorProb = findViewById(R.id.tvVersicolorProb);
        tvVirginicaProb = findViewById(R.id.tvVirginicaProb);

        progressSetosa = findViewById(R.id.progressSetosa);
        progressVersicolor = findViewById(R.id.progressVersicolor);
        progressVirginica = findViewById(R.id.progressVirginica);
    }

    private void initClassifier() {
        executor.execute(() -> {
            try {
                classifier = new IrisClassifier(this, "rf_iris.onnx");
                runOnUiThread(this::runPrediction);
            } catch (Exception e) {
                Log.e(TAG, "Failed to load ONNX model", e);
                runOnUiThread(() -> Toast.makeText(this, "Error loading model", Toast.LENGTH_SHORT).show());
            }
        });
    }

    private void setupListeners() {
        sliderSepalLength.addOnChangeListener((slider, value, fromUser) -> {
            tvSepalLengthValue.setText(String.format(Locale.US, "%.1f cm", value));
            runPrediction();
        });

        sliderSepalWidth.addOnChangeListener((slider, value, fromUser) -> {
            tvSepalWidthValue.setText(String.format(Locale.US, "%.1f cm", value));
            runPrediction();
        });

        sliderPetalLength.addOnChangeListener((slider, value, fromUser) -> {
            tvPetalLengthValue.setText(String.format(Locale.US, "%.1f cm", value));
            runPrediction();
        });

        sliderPetalWidth.addOnChangeListener((slider, value, fromUser) -> {
            tvPetalWidthValue.setText(String.format(Locale.US, "%.1f cm", value));
            runPrediction();
        });

        btnPredict.setOnClickListener(v -> runPrediction());
    }

    private void runPrediction() {
        if (classifier == null) return;

        float sepalLength = sliderSepalLength.getValue();
        float sepalWidth = sliderSepalWidth.getValue();
        float petalLength = sliderPetalLength.getValue();
        float petalWidth = sliderPetalWidth.getValue();

        float[] features = {sepalLength, sepalWidth, petalLength, petalWidth};

        executor.execute(() -> {
            try {
                IrisClassifier.Result result = classifier.predict(features);
                runOnUiThread(() -> displayResult(result));
            } catch (Exception e) {
                Log.e(TAG, "Prediction error", e);
            }
        });
    }

    private void displayResult(IrisClassifier.Result result) {
        String className = result.className();
        String formattedClass = className.substring(0, 1).toUpperCase(Locale.US) + className.substring(1);
        tvPredictedClass.setText(String.format(Locale.US, "Predicted: %s", formattedClass));

        float[] probs = result.probabilities;
        if (probs != null && probs.length >= 3) {
            float pSetosa = probs[0] * 100f;
            float pVersicolor = probs[1] * 100f;
            float pVirginica = probs[2] * 100f;

            tvSetosaProb.setText(String.format(Locale.US, "Setosa: %.1f%%", pSetosa));
            progressSetosa.setProgress((int) pSetosa);

            tvVersicolorProb.setText(String.format(Locale.US, "Versicolor: %.1f%%", pVersicolor));
            progressVersicolor.setProgress((int) pVersicolor);

            tvVirginicaProb.setText(String.format(Locale.US, "Virginica: %.1f%%", pVirginica));
            progressVirginica.setProgress((int) pVirginica);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(() -> {
            if (classifier != null) {
                try {
                    classifier.close();
                } catch (Exception e) {
                    Log.e(TAG, "Error closing classifier", e);
                }
            }
        });
        executor.shutdown();
    }
}
