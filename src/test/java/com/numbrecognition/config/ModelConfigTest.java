package com.numbrecognition.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests für die Modell-Konfigurationskonstanten.
 */
class ModelConfigTest {

    @Test
    void imageDimensions_are28x28() {
        assertEquals(28, ModelConfig.IMAGE_HEIGHT);
        assertEquals(28, ModelConfig.IMAGE_WIDTH);
    }

    @Test
    void numClasses_is10() {
        assertEquals(10, ModelConfig.NUM_CLASSES);
    }

    @Test
    void epochs_arePositive() {
        assertTrue(ModelConfig.EPOCHS > 0, "Epochen müssen größer als 0 sein");
    }

    @Test
    void batchSize_isPositive() {
        assertTrue(ModelConfig.BATCH_SIZE > 0, "Batch-Größe muss größer als 0 sein");
    }

    @Test
    void learningRate_isPositive() {
        assertTrue(ModelConfig.LEARNING_RATE > 0, "Lernrate muss größer als 0 sein");
    }

    @Test
    void cnnArchitecture_paramsAreValid() {
        assertTrue(ModelConfig.CONV1_FILTERS > 0);
        assertTrue(ModelConfig.CONV2_FILTERS > 0);
        assertTrue(ModelConfig.DENSE_UNITS > 0);
        assertTrue(ModelConfig.DROPOUT_RATE >= 0.0 && ModelConfig.DROPOUT_RATE < 1.0);
    }

    @Test
    void modelFile_hasValidName() {
        assertNotNull(ModelConfig.MODEL_FILE);
        assertFalse(ModelConfig.MODEL_FILE.isEmpty());
        assertTrue(ModelConfig.MODEL_FILE.endsWith(".zip"));
    }
}
