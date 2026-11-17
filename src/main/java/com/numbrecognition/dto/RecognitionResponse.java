package com.numbrecognition.dto;

import java.util.List;

/**
 * Antwort für Ziffernerkennung
 */
public class RecognitionResponse {

    private int digit;
    private double confidence;
    private List<Double> probabilities;

    public RecognitionResponse() {
    }

    public RecognitionResponse(int digit, double confidence, List<Double> probabilities) {
        this.digit = digit;
        this.confidence = confidence;
        this.probabilities = probabilities;
    }

    public int getDigit() {
        return digit;
    }

    public void setDigit(int digit) {
        this.digit = digit;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }

    public List<Double> getProbabilities() {
        return probabilities;
    }

    public void setProbabilities(List<Double> probabilities) {
        this.probabilities = probabilities;
    }
}
