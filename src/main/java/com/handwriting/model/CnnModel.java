package com.handwriting.model;

import com.handwriting.config.ModelConfig;
import com.handwriting.data.MnistData;
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

import java.io.File;
import java.util.Random;

/**
 * Convolutional Neural Network für MNIST-Handschrifterkennung
 * <p>
 * Architektur:
 * Input (28x28) → Conv2D(32) → ReLU → MaxPooling → Conv2D(64) → ReLU → MaxPooling
 * → Dense(128) → ReLU → Dropout(0.5) → Output(10) → Softmax
 */
public class CnnModel {

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
        System.out.println("=== CNN Training gestartet ===");
        System.out.println("Epochen: " + ModelConfig.EPOCHS);
        System.out.println("Batch Size: " + ModelConfig.BATCH_SIZE);
        System.out.println("Training Samples: " + trainingData.size());

        network = new MultiLayerNetwork(buildConfiguration());
        network.init();

        INDArray images = trainingData.getImages();
        INDArray labels = trainingData.getLabels();
        int numSamples = trainingData.size();

        for (int epoch = 0; epoch < ModelConfig.EPOCHS; epoch++) {
            // Mini-Batch Training mit direkten Slices (schneller als INDArrayIndex.put)
            for (int i = 0; i < numSamples; i += ModelConfig.BATCH_SIZE) {
                int end = Math.min(i + ModelConfig.BATCH_SIZE, numSamples);
                INDArray batchImages = images.get(new INDArrayIndex[]{NDArrayIndex.interval(i, end), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()});
                INDArray batchLabels = labels.get(new INDArrayIndex[]{NDArrayIndex.interval(i, end), NDArrayIndex.all()});

                network.fit(batchImages, batchLabels);
            }

            Evaluation eval = evaluate(testData);
            System.out.println("Epoch " + (epoch + 1) + "/" + ModelConfig.EPOCHS + " — Accuracy: " + String.format("%.2f%%", eval.accuracy() * 100));
        }

        testAccuracy = evaluate(testData).accuracy();
        epochsTrained = ModelConfig.EPOCHS;

        System.out.println("=== Training abgeschlossen ===");
        System.out.println("Test Accuracy: " + String.format("%.2f%%", testAccuracy * 100));
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
            System.out.println("Modell gespeichert: " + filePath);
        } catch (Exception e) {
            System.out.println("Fehler beim Speichern: " + e.getMessage());
        }
    }

    /**
     * Modell laden und Accuracy durch kurzes Evaluation berechnen.
     * Da DeepLearning4J die Accuracy nicht in der model.zip speichert,
     * wird sie hier mit den Testdaten neu berechnet (~1 Sekunde).
     */
    public boolean load(String filePath, MnistData testData) {
        File modelFile = new File(filePath);
        if (!modelFile.exists()) {
            System.out.println("Modell-Datei nicht gefunden: " + filePath);
            return false;
        }
        try {
            network = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            // Accuracy neu berechnen, da sie nicht in der Datei gespeichert ist
            if (testData != null) {
                testAccuracy = evaluate(testData).accuracy();
            } else {
                testAccuracy = 0.0;
            }
            epochsTrained = ModelConfig.EPOCHS;
            System.out.println("Modell geladen: " + filePath + " (Accuracy: " + String.format("%.2f%%", testAccuracy * 100) + ")");
            return true;
        } catch (Exception e) {
            System.out.println("Fehler beim Laden: " + e.getClass().getSimpleName() + " - " + e.getMessage());
            System.out.println("Die model.zip wird NICHT gelöscht. Bitte stelle sicher, dass die Modell-Architektur im Code zur gespeicherten Datei passt.");
            return false;
        }
    }

    public MultiLayerNetwork getNetwork() { return network; }
    public double getTestAccuracy() { return testAccuracy; }
    public int getEpochsTrained() { return epochsTrained; }
    public boolean isLoaded() { return network != null; }
}
