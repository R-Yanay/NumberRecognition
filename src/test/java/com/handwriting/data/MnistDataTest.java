package com.handwriting.data;

import com.handwriting.config.ModelConfig;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests für den MnistData-Container.
 */
class MnistDataTest {

    @Test
    void size_returnsCorrectCount() {
        // Erstelle Dummy-Daten mit 5 Samples
        int samples = 5;
        org.nd4j.linalg.api.ndarray.INDArray images = Nd4j.zeros(samples, 1, 28, 28);
        org.nd4j.linalg.api.ndarray.INDArray labels = Nd4j.zeros(samples, 10);

        MnistData data = new MnistData(images, labels, "test");

        assertEquals(samples, data.size());
    }

    @Test
    void constructor_setsFieldsCorrectly() {
        int samples = 3;
        org.nd4j.linalg.api.ndarray.INDArray images = Nd4j.zeros(samples, 1, 28, 28);
        org.nd4j.linalg.api.ndarray.INDArray labels = Nd4j.zeros(samples, 10);

        MnistData data = new MnistData(images, labels, "training");

        assertEquals(images, data.getImages());
        assertEquals(labels, data.getLabels());
        assertEquals("training", data.getName());
    }

    @Test
    void toString_containsNameAndSize() {
        int samples = 10;
        org.nd4j.linalg.api.ndarray.INDArray images = Nd4j.zeros(samples, 1, 28, 28);
        org.nd4j.linalg.api.ndarray.INDArray labels = Nd4j.zeros(samples, 10);

        MnistData data = new MnistData(images, labels, "test");

        String str = data.toString();
        assertTrue(str.contains("test"));
        assertTrue(str.contains("10"));
    }

    @Test
    void images_shapeIsCorrect() {
        int samples = 2;
        org.nd4j.linalg.api.ndarray.INDArray images = Nd4j.zeros(samples, 1, 28, 28);
        org.nd4j.linalg.api.ndarray.INDArray labels = Nd4j.zeros(samples, 10);

        MnistData data = new MnistData(images, labels, "test");

        long[] shape = data.getImages().shape();
        assertEquals(samples, shape[0]);
        assertEquals(1, shape[1]);
        assertEquals(ModelConfig.IMAGE_HEIGHT, shape[2]);
        assertEquals(ModelConfig.IMAGE_WIDTH, shape[3]);
    }

    @Test
    void labels_shapeIsCorrect() {
        int samples = 2;
        org.nd4j.linalg.api.ndarray.INDArray images = Nd4j.zeros(samples, 1, 28, 28);
        org.nd4j.linalg.api.ndarray.INDArray labels = Nd4j.zeros(samples, 10);

        MnistData data = new MnistData(images, labels, "test");

        long[] shape = data.getLabels().shape();
        assertEquals(samples, shape[0]);
        assertEquals(ModelConfig.NUM_CLASSES, shape[1]);
    }
}
