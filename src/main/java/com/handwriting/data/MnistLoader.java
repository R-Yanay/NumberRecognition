package com.handwriting.data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.net.URL;
import java.util.zip.GZIPInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Lädt den MNIST-Datensatz (70.000 handschriftliche Ziffern 0-9)
 * von http://yann.lecun.com/exdb/mnist/
 * <p>
 * MNIST Format (IDX-Files):
 * - Magic Number (4 bytes)
 * - Dimensions (4 bytes pro Dimension)
 * - Data (rest)
 */
public class MnistLoader {

    private static final String BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/";
    private static final String TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS = "t10k-labels-idx1-ubyte.gz";

    private static final int IMAGE_HEIGHT = 28;
    private static final int IMAGE_WIDTH = 28;
    private static final int NUM_CLASSES = 10;

    /**
     * Lädt den MNIST-Trainingsdatensatz (60.000 Bilder)
     */
    public MnistData loadTrainingData() throws IOException {
        byte[] imagesBytes = downloadAndDecompress(BASE_URL + TRAIN_IMAGES);
        byte[] labelsBytes = downloadAndDecompress(BASE_URL + TRAIN_LABELS);

        int numImages = parseIntBigEndian(labelsBytes, 0) == 2049 ? parseIntBigEndian(labelsBytes, 4) : 60000;

        INDArray images = parseImages(imagesBytes, numImages);
        INDArray labels = parseLabels(labelsBytes, numImages);

        return new MnistData(images, labels, "training");
    }

    /**
     * Lädt den MNIST-Testdatensatz (10.000 Bilder)
     */
    public MnistData loadTestData() throws IOException {
        byte[] imagesBytes = downloadAndDecompress(BASE_URL + TEST_IMAGES);
        byte[] labelsBytes = downloadAndDecompress(BASE_URL + TEST_LABELS);

        int numImages = parseIntBigEndian(labelsBytes, 0) == 2049 ? parseIntBigEndian(labelsBytes, 4) : 10000;

        INDArray images = parseImages(imagesBytes, numImages);
        INDArray labels = parseLabels(labelsBytes, numImages);

        return new MnistData(images, labels, "test");
    }

    /**
     * Download und GZIP-Dekompression
     */
    private byte[] downloadAndDecompress(String urlStr) throws IOException {
        System.out.println("Downloading: " + urlStr);
        URL url = new URL(urlStr);

        try (InputStream in = new GZIPInputStream(url.openStream());
             ByteArrayOutputStream buffer = new ByteArrayOutputStream()) {

            int read;
            byte[] data = new byte[4096];
            while ((read = in.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, read);
            }
            return buffer.toByteArray();
        }
    }

    /**
     * Parse IDX-Image-File zu INDArray
     * Format: magic(4) + numImages(4) + rows(4) + cols(4) + data
     */
    private INDArray parseImages(byte[] data, int numImages) {
        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.BIG_ENDIAN);
        buffer.getInt(); // magic number
        buffer.getInt(); // num images
        buffer.getInt(); // rows
        buffer.getInt(); // cols

        // Flaches Array: [numImages * 1 * 28 * 28]
        float[] flat = new float[numImages * IMAGE_HEIGHT * IMAGE_WIDTH];

        for (int i = 0; i < numImages; i++) {
            for (int row = 0; row < IMAGE_HEIGHT; row++) {
                for (int col = 0; col < IMAGE_WIDTH; col++) {
                    // Normalisieren: 0-255 → 0.0-1.0 (wichtig: & 0xFF für unsigned byte)
                    flat[i * IMAGE_HEIGHT * IMAGE_WIDTH + row * IMAGE_WIDTH + col] = (buffer.get() & 0xFF) / 255.0f;
                }
            }
        }

        // Shape: [numImages, 1, 28, 28] (batch, channels, height, width)
        INDArray images = Nd4j.create(flat);
        return images.reshape('c', numImages, 1, IMAGE_HEIGHT, IMAGE_WIDTH);
    }

    /**
     * Parse IDX-Label-File zu INDArray (One-Hot-Encoding)
     * Format: magic(4) + numLabels(4) + labels
     */
    private INDArray parseLabels(byte[] data, int numLabels) {
        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.BIG_ENDIAN);
        buffer.getInt(); // magic number
        buffer.getInt(); // num labels

        // One-Hot-Encoding: [7, 0, 0, 0, 0, 0, 0, 0, 0, 0] für Label 7
        float[][] labels = new float[numLabels][NUM_CLASSES];

        for (int i = 0; i < numLabels; i++) {
            int label = buffer.get() & 0xFF;
            labels[i][label] = 1.0f;
        }

        return Nd4j.create(labels);
    }

    /**
     * Helper: Big-Endian Integer parsen
     */
    private int parseIntBigEndian(byte[] data, int offset) {
        return ((data[offset] & 0xFF) << 24) |
                ((data[offset + 1] & 0xFF) << 16) |
                ((data[offset + 2] & 0xFF) << 8) |
                (data[offset + 3] & 0xFF);
    }
}
