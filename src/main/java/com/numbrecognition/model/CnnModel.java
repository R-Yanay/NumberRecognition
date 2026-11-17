package com.numbrecognition.model;

import com.numbrecognition.config.ModelConfig;
import com.numbrecognition.data.MnistData;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Convolutional Neural Network für MNIST-Handschrifterkennung.
 * <p>
 * Architektur:
 * Input (28×28) → Conv2D(32) → ReLU → MaxPooling → Conv2D(64) → ReLU → MaxPooling
 * → Dense(128) → ReLU → Dropout(0.5) → Output(10) → Softmax
 */
public class CnnModel {

    private static final Logger log = LoggerFactory.getLogger(CnnModel.class);

    private MultiLayerNetwork network;
    private double testAccuracy;
    private int epochsTrained;

    /**
     * CNN-Architektur definieren
     */
    public MultiLayerConfiguration buildConfiguration() {
        return new NeuralNetConfiguration.Builder()
                .seed(42)
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()

                // Layer 1: Convolution (32 Filter, 5x5 Kernel)
                .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5})
                        .nIn(1)
                        .nOut(ModelConfig.CONV1_FILTERS)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())

                // Layer 2: MaxPooling (2x2)
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())

                // Layer 3: Convolution (64 Filter, 5x5 Kernel)
                .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5})
                        .nOut(ModelConfig.CONV2_FILTERS)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())

                // Layer 4: MaxPooling (2x2)
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())

                // Layer 5: Dense (128 Neuronen)
                .layer(4, new DenseLayer.Builder()
                        .nOut(ModelConfig.DENSE_UNITS)
                        .activation(Activation.RELU)
                        .build())

                // Layer 6: Dropout (0.5)
                .layer(5, new DropoutLayer.Builder(ModelConfig.DROPOUT_RATE).build())

                // Layer 7: Output (10 Klassen, Softmax)
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(ModelConfig.NUM_CLASSES)
                        .activation(Activation.SOFTMAX)
                        .build())

                .setInputType(org.deeplearning4j.nn.conf.inputs.InputType.convolutional(
                        ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH, 1))
                .build();
    }

    /**
     * Modell trainieren mit MNIST-Daten
     */
    public void train(MnistData trainingData, MnistData testData) {
        log.info("=== CNN Training gestartet ===");
        log.info("Epochen: {}, Batch Size: {}, Training Samples: {}",
                ModelConfig.EPOCHS, ModelConfig.BATCH_SIZE, trainingData.size());

        network = new MultiLayerNetwork(buildConfiguration());
        network.init();

        INDArray images = trainingData.getImages();
        INDArray labels = trainingData.getLabels();
        int numSamples = trainingData.size();

        for (int epoch = 0; epoch < ModelConfig.EPOCHS; epoch++) {
            // Mini-Batch Training mit direkten Slices
            for (int i = 0; i < numSamples; i += ModelConfig.BATCH_SIZE) {
                int end = Math.min(i + ModelConfig.BATCH_SIZE, numSamples);
                INDArray batchImages = images.get(new INDArrayIndex[]{
                        NDArrayIndex.interval(i, end), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all()});
                INDArray batchLabels = labels.get(new INDArrayIndex[]{
                        NDArrayIndex.interval(i, end), NDArrayIndex.all()});

                network.fit(batchImages, batchLabels);
            }

            Evaluation eval = evaluate(testData);
            log.info("Epoch {}/{} — Accuracy: {}%",
                    epoch + 1, ModelConfig.EPOCHS, String.format("%.2f", eval.accuracy() * 100));
        }

        testAccuracy = evaluate(testData).accuracy();
        epochsTrained = ModelConfig.EPOCHS;

        log.info("=== Training abgeschlossen ===");
        log.info("Test Accuracy: {}%", String.format("%.2f", testAccuracy * 100));
    }

    /**
     * Modell evaluieren (Accuracy berechnen)
     */
    public Evaluation evaluate(MnistData testData) {
        Evaluation eval = new Evaluation(ModelConfig.NUM_CLASSES);
        INDArray predictions = network.output(testData.getImages(), false);
        eval.eval(testData.getLabels(), predictions);
        return eval;
    }

    /**
     * Einzelnes Bild erkennen (Inferenz)
     */
    public PredictionResult predict(INDArray image) {
        if (network == null) {
            throw new IllegalStateException("Modell nicht geladen.");
        }

        INDArray output = network.output(image, false);
        int predictedDigit = output.argMax(1).getInt(0, 0);
        double confidence = output.getDouble(0, predictedDigit);

        return new PredictionResult(predictedDigit, confidence, output);
    }

    /**
     * Modell speichern
     */
    public void save(String filePath) {
        if (network == null) throw new IllegalStateException("Kein Modell zum Speichern.");
        try {
            ModelSerializer.writeModel(network, new File(filePath), true);
            log.info("Modell gespeichert: {}", filePath);
        } catch (Exception e) {
            log.error("Fehler beim Speichern: {}", e.getMessage());
        }
    }

    /**
     * Modell laden und Accuracy durch Evaluation berechnen.
     * Da DeepLearning4J die Accuracy nicht in der model.zip speichert,
     * wird sie hier mit den Testdaten neu berechnet (~1 Sekunde).
     */
    public boolean load(String filePath, MnistData testData) {
        File modelFile = new File(filePath);
        if (!modelFile.exists()) {
            log.info("Modell-Datei nicht gefunden: {}", filePath);
            return false;
        }
        try {
            network = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            if (testData != null) {
                testAccuracy = evaluate(testData).accuracy();
            } else {
                testAccuracy = 0.0;
            }
            epochsTrained = ModelConfig.EPOCHS;
            log.info("Modell geladen: {} (Accuracy: {}%)",
                    filePath, String.format("%.2f", testAccuracy * 100));
            return true;
        } catch (Exception e) {
            log.warn("Fehler beim Laden: {} — {}. Stelle sicher, dass die Modell-Architektur zur gespeicherten Datei passt.",
                    e.getClass().getSimpleName(), e.getMessage());
            return false;
        }
    }

    public MultiLayerNetwork getNetwork() { return network; }
    public double getTestAccuracy() { return testAccuracy; }
    public int getEpochsTrained() { return epochsTrained; }
    public boolean isLoaded() { return network != null; }
}
