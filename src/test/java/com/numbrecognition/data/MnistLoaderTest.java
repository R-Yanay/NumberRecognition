package com.numbrecognition.data;

import com.numbrecognition.config.ModelConfig;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.GZIPOutputStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests für den MnistLoader.
 * Da der Loader Internet benötigt, werden hier synthetische IDX-Daten getestet.
 */
class MnistLoaderTest {

    /**
     * Erstelle synthetische MNIST-Image-Daten im IDX-Format (GZIP-komprimiert).
     */
    private byte[] createSyntheticImages(int numImages) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ByteBuffer buffer = ByteBuffer.allocate(16 + numImages * 28 * 28);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putInt(2051); // magic number
        buffer.putInt(numImages);
        buffer.putInt(28); // rows
        buffer.putInt(28); // cols
        for (int i = 0; i < numImages; i++) {
            for (int row = 0; row < 28; row++) {
                for (int col = 0; col < 28; col++) {
                    buffer.put((byte) 200); // heller Pixelwert
                }
            }
        }
        byte[] idxData = buffer.array();

        // GZIP-komprimieren
        try (GZIPOutputStream gzos = new GZIPOutputStream(baos)) {
            gzos.write(idxData);
        }
        return baos.toByteArray();
    }

    /**
     * Erstelle synthetische MNIST-Label-Daten im IDX-Format (GZIP-komprimiert).
     */
    private byte[] createSyntheticLabels(int numLabels) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ByteBuffer buffer = ByteBuffer.allocate(8 + numLabels);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putInt(2049); // magic number
        buffer.putInt(numLabels);
        for (int i = 0; i < numLabels; i++) {
            buffer.put((byte) (i % 10)); // Labels 0-9 zyklisch
        }
        byte[] idxData = buffer.array();

        // GZIP-komprimieren
        try (GZIPOutputStream gzos = new GZIPOutputStream(baos)) {
            gzos.write(idxData);
        }
        return baos.toByteArray();
    }

    @Test
    void parseImages_normalizesPixelValues() throws Exception {
        // Testet, dass Pixelwerte korrekt normalisiert werden (0–255 → 0.0–1.0)
        MnistLoader loader = new MnistLoader();
        java.lang.reflect.Method method = MnistLoader.class.getDeclaredMethod("parseImages", byte[].class, int.class);
        method.setAccessible(true);

        // Erstelle IDX-Daten OHNE GZIP (parseImages erwartet rohe IDX-Bytes)
        int numImages = 2;
        ByteBuffer buffer = ByteBuffer.allocate(16 + numImages * 28 * 28);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putInt(2051); // magic number
        buffer.putInt(numImages);
        buffer.putInt(28); // rows
        buffer.putInt(28); // cols
        for (int i = 0; i < numImages * 28 * 28; i++) {
            buffer.put((byte) 200);
        }
        byte[] imagesBytes = buffer.array();

        INDArray images = (INDArray) method.invoke(loader, imagesBytes, numImages);

        // Shape prüfen
        long[] shape = images.shape();
        assertEquals(2, shape[0]);
        assertEquals(1, shape[1]);
        assertEquals(28, shape[2]);
        assertEquals(28, shape[3]);

        // Pixelwerte prüfen (200/255.0 ≈ 0.784)
        double pixel = images.getDouble(0, 0, 0, 0);
        assertEquals(200.0 / 255.0, pixel, 0.001);
    }

    @Test
    void parseLabels_createsOneHotEncoding() throws Exception {
        MnistLoader loader = new MnistLoader();
        java.lang.reflect.Method method = MnistLoader.class.getDeclaredMethod("parseLabels", byte[].class, int.class);
        method.setAccessible(true);

        // Erstelle IDX-Daten OHNE GZIP (parseLabels erwartet rohe IDX-Bytes)
        int numLabels = 5;
        ByteBuffer buffer = ByteBuffer.allocate(8 + numLabels);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putInt(2049); // magic number
        buffer.putInt(numLabels);
        for (int i = 0; i < numLabels; i++) {
            buffer.put((byte) (i % 10));
        }
        byte[] labelsBytes = buffer.array();

        INDArray labels = (INDArray) method.invoke(loader, labelsBytes, numLabels);

        // Shape prüfen
        long[] shape = labels.shape();
        assertEquals(5, shape[0]);
        assertEquals(ModelConfig.NUM_CLASSES, shape[1]);

        // One-Hot-Encoding prüfen: Label 0 → [1,0,0,...], Label 1 → [0,1,0,...]
        assertEquals(1.0, labels.getDouble(0, 0), 0.001); // Label 0
        assertEquals(1.0, labels.getDouble(1, 1), 0.001); // Label 1
        assertEquals(1.0, labels.getDouble(2, 2), 0.001); // Label 2
    }

    @Test
    void unsignedByteMasking_handlesHighPixelValues() throws Exception {
        // Testet den kritischen & 0xFF Bug: Pixel > 127 dürfen nicht negativ sein
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ByteBuffer buffer = ByteBuffer.allocate(16 + 1 * 28 * 28);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putInt(2051);
        buffer.putInt(1);
        buffer.putInt(28);
        buffer.putInt(28);
        for (int i = 0; i < 28 * 28; i++) {
            buffer.put((byte) 255); // höchster Pixelwert (als signed byte: -1)
        }
        byte[] idxData = buffer.array();
        try (GZIPOutputStream gzos = new GZIPOutputStream(baos)) {
            gzos.write(idxData);
        }

        MnistLoader loader = new MnistLoader();
        java.lang.reflect.Method method = MnistLoader.class.getDeclaredMethod("parseImages", byte[].class, int.class);
        method.setAccessible(true);
        INDArray images = (INDArray) method.invoke(loader, idxData, 1);

        // Alle Pixel müssen 1.0 sein (255/255.0), nicht negativ
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                double pixel = images.getDouble(0, 0, row, col);
                assertTrue(pixel >= 0.0, "Pixel darf nicht negativ sein: " + pixel);
                assertEquals(1.0, pixel, 0.001);
            }
        }
    }
}
