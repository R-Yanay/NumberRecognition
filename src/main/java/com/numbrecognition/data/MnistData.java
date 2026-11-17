package com.numbrecognition.data;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Container für MNIST-Daten (Bilder + Labels)
 */
public class MnistData {

    private final INDArray images;    // Shape: [numSamples, 1, 28, 28]
    private final INDArray labels;    // Shape: [numSamples, 10] (one-hot)
    private final String name;

    public MnistData(INDArray images, INDArray labels, String name) {
        this.images = images;
        this.labels = labels;
        this.name = name;
    }

    public INDArray getImages() {
        return images;
    }

    public INDArray getLabels() {
        return labels;
    }

    public String getName() {
        return name;
    }

    public int size() {
        return (int) images.size(0);
    }

    @Override
    public String toString() {
        return "MnistData{name='" + name + "', samples=" + size() + "}";
    }
}
