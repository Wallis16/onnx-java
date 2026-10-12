package com.example.irisonnx;

import android.content.Context;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxMap;
import ai.onnxruntime.OnnxSequence;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * Runs the scikit-learn RandomForest (Iris) model exported with skl2onnx.
 *
 * Model signature:
 *   input  "X"                  float32 [N, 4]  (sepal length, sepal width, petal length, petal width)
 *   output "output_label"       int64   [N]
 *   output "output_probability" seq(map(int64, float))  (ZipMap)
 *
 * Create once, reuse for many predictions, call close() when done.
 */
public class IrisClassifier implements AutoCloseable {

    public static final int NUM_FEATURES = 4;
    public static final String[] CLASS_NAMES = {"setosa", "versicolor", "virginica"};

    private static final String LABEL_OUTPUT = "output_label";
    private static final String PROB_OUTPUT = "output_probability";

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    public IrisClassifier(Context context, String assetFileName) throws IOException, OrtException {
        env = OrtEnvironment.getEnvironment();
        byte[] modelBytes = readAsset(context, assetFileName);
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            session = env.createSession(modelBytes, options);
        }
        inputName = session.getInputNames().iterator().next(); // "X"
    }

    /** Classifies one sample of 4 features. */
    public Result predict(float[] features) throws OrtException {
        if (features == null || features.length != NUM_FEATURES) {
            throw new IllegalArgumentException("Expected " + NUM_FEATURES + " features");
        }

        long[] shape = {1, NUM_FEATURES};
        try (OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(features), shape);
             OrtSession.Result output = session.run(Collections.singletonMap(inputName, input))) {

            long[] labels = (long[]) output.get(LABEL_OUTPUT)
                    .orElseThrow(() -> new IllegalStateException("Missing " + LABEL_OUTPUT))
                    .getValue();
            int label = (int) labels[0];

            float[] probabilities = new float[CLASS_NAMES.length];
            OnnxValue probValue = output.get(PROB_OUTPUT).orElse(null);
            if (probValue instanceof OnnxSequence) {
                List<?> rows = ((OnnxSequence) probValue).getValue();
                Map<?, ?> row = toMap(rows.get(0));
                for (Map.Entry<?, ?> e : row.entrySet()) {
                    int cls = ((Number) e.getKey()).intValue();
                    if (cls >= 0 && cls < probabilities.length) {
                        probabilities[cls] = ((Number) e.getValue()).floatValue();
                    }
                }
            }
            return new Result(label, probabilities);
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
        // OrtEnvironment is a process-wide singleton; don't close it here.
    }

    // Depending on the onnxruntime version, sequence elements are OnnxMap or plain Map.
    private static Map<?, ?> toMap(Object element) throws OrtException {
        if (element instanceof OnnxMap) {
            return ((OnnxMap) element).getValue();
        }
        return (Map<?, ?>) element;
    }

    private static byte[] readAsset(Context context, String name) throws IOException {
        try (InputStream in = context.getAssets().open(name);
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) != -1) {
                out.write(buf, 0, n);
            }
            return out.toByteArray();
        }
    }

    public static final class Result {
        public final int label;
        public final float[] probabilities;

        Result(int label, float[] probabilities) {
            this.label = label;
            this.probabilities = probabilities;
        }

        public String className() {
            return label >= 0 && label < CLASS_NAMES.length ? CLASS_NAMES[label] : String.valueOf(label);
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder("label=").append(label)
                    .append(" (").append(className()).append(") probs=[");
            for (int i = 0; i < probabilities.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(CLASS_NAMES[i]).append('=').append(String.format("%.3f", probabilities[i]));
            }
            return sb.append(']').toString();
        }
    }
}
