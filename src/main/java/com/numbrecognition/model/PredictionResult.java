package com.numbrecognition.model;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Ergebnis einer Ziffernerkennung
 */
public class PredictionResult {

    private final int digit;
    private final double confidence;
    private final INDArray probabilities;

    public PredictionResult(int digit, double confidence, INDArray probabilities) {
        this.digit = digit;
        this.confidence = confidence;
        this.probabilities = probabilities;
    }

    public int getDigit() {
        return digit;
    }

    public double getConfidence() {
        return confidence;
    }

    public INDArray getProbabilities() {
        return probabilities;
    }
}
