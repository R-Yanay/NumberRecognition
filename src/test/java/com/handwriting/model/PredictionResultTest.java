package com.handwriting.model;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests für das PredictionResult-DTO.
 */
class PredictionResultTest {

    @Test
    void constructor_setsFieldsCorrectly() {
        double[] probs = {0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9};
        org.nd4j.linalg.api.ndarray.INDArray probArray = Nd4j.create(probs);

        PredictionResult result = new PredictionResult(7, 0.9, probArray);

        assertEquals(7, result.getDigit());
        assertEquals(0.9, result.getConfidence(), 0.001);
        assertNotNull(result.getProbabilities());
        assertEquals(10, result.getProbabilities().shape()[0]);
    }

    @Test
    void getters_returnCorrectValues() {
        double[] probs = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        org.nd4j.linalg.api.ndarray.INDArray probArray = Nd4j.create(probs);

        PredictionResult result = new PredictionResult(3, 1.0, probArray);

        assertEquals(3, result.getDigit());
        assertEquals(1.0, result.getConfidence(), 0.001);
        assertEquals(1.0, result.getProbabilities().getDouble(3), 0.001);
    }

    @Test
    void probabilities_sumToOne() {
        // 9 * 0.04 + 0.64 = 1.0
        double[] probs = {0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.64};
        org.nd4j.linalg.api.ndarray.INDArray probArray = Nd4j.create(probs);

        PredictionResult result = new PredictionResult(9, 0.64, probArray);

        double sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += result.getProbabilities().getDouble(i);
        }
        assertEquals(1.0, sum, 0.01, "Wahrscheinlichkeiten müssen sich zu 1.0 summieren");
    }
}
