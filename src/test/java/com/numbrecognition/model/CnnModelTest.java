package com.numbrecognition.model;

import com.numbrecognition.config.ModelConfig;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests für das CnnModel.
 * Testet Architektur, Vorhersage und Speichern/Laden.
 */
class CnnModelTest {

    @TempDir
    Path tempDir;

    @Test
    void buildConfiguration_createsValidConfig() {
        CnnModel model = new CnnModel();
        MultiLayerConfiguration config = model.buildConfiguration();

        assertNotNull(config);
        assertEquals(7, config.getConfs().size(), "CNN muss 7 Layer haben");
    }

    @Test
    void buildConfiguration_hasCorrectLayerCount() {
        CnnModel model = new CnnModel();
        MultiLayerConfiguration config = model.buildConfiguration();

        // Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(128) → Dropout → Output(10)
        assertEquals(7, config.getConfs().size());
    }

    @Test
    void predict_returnsValidProbabilities() {
        CnnModel model = new CnnModel();
        MultiLayerNetwork network = new MultiLayerNetwork(model.buildConfiguration());
        network.init();
        model = new CnnModel();
        // Setze network via Reflection, da kein öffentlicher Setter
        try {
            java.lang.reflect.Field field = CnnModel.class.getDeclaredField("network");
            field.setAccessible(true);
            field.set(model, network);
        } catch (Exception e) {
            fail("Reflection fehlgeschlagen: " + e.getMessage());
        }

        // Erstelle ein Dummy-Bild [1, 1, 28, 28]
        org.nd4j.linalg.api.ndarray.INDArray image = Nd4j.zeros(1, 1, 28, 28);

        PredictionResult result = model.predict(image);

        assertNotNull(result);
        assertTrue(result.getDigit() >= 0 && result.getDigit() <= 9);
        assertTrue(result.getConfidence() > 0);
        assertNotNull(result.getProbabilities());
    }

    @Test
    void predict_throwsWhenModelNotLoaded() {
        CnnModel model = new CnnModel();
        org.nd4j.linalg.api.ndarray.INDArray image = Nd4j.zeros(1, 1, 28, 28);

        assertThrows(IllegalStateException.class, () -> model.predict(image));
    }

    @Test
    void isLoaded_returnsFalseInitially() {
        CnnModel model = new CnnModel();
        assertFalse(model.isLoaded());
    }

    @Test
    void isLoaded_returnsTrueAfterInit() {
        CnnModel model = new CnnModel();
        MultiLayerNetwork network = new MultiLayerNetwork(model.buildConfiguration());
        network.init();

        try {
            java.lang.reflect.Field field = CnnModel.class.getDeclaredField("network");
            field.setAccessible(true);
            field.set(model, network);
        } catch (Exception e) {
            fail("Reflection fehlgeschlagen: " + e.getMessage());
        }

        assertTrue(model.isLoaded());
    }

    @Test
    void save_and_load_restoresModel(@TempDir Path tempDir) throws Exception {
        CnnModel model = new CnnModel();
        MultiLayerNetwork network = new MultiLayerNetwork(model.buildConfiguration());
        network.init();

        java.lang.reflect.Field field = CnnModel.class.getDeclaredField("network");
        field.setAccessible(true);
        field.set(model, network);

        // Speichere das Modell
        String filePath = tempDir.resolve("test-model.zip").toString();
        model.save(filePath);

        assertTrue(new File(filePath).exists());

        // Lade das Modell in ein neues CnnModel
        CnnModel loadedModel = new CnnModel();
        boolean loaded = loadedModel.load(filePath, null);

        assertTrue(loaded, "Modell sollte erfolgreich geladen werden");
        assertTrue(loadedModel.isLoaded());
    }

    @Test
    void load_nonExistentFile_returnsFalse() {
        CnnModel model = new CnnModel();
        boolean loaded = model.load("nicht-existent.zip", null);
        assertFalse(loaded);
    }

    @Test
    void load_corruptedFile_returnsFalse(@TempDir Path tempDir) throws Exception {
        // Erstelle eine beschädigte Datei
        File corruptedFile = tempDir.resolve("corrupted.zip").toFile();
        corruptedFile.createNewFile();

        CnnModel model = new CnnModel();
        boolean loaded = model.load(corruptedFile.getAbsolutePath(), null);
        assertFalse(loaded, "Beschädigte Datei sollte nicht geladen werden");
    }

    @Test
    void getTestAccuracy_returnsZeroInitially() {
        CnnModel model = new CnnModel();
        assertEquals(0.0, model.getTestAccuracy(), 0.001);
    }

    @Test
    void getEpochsTrained_returnsZeroInitially() {
        CnnModel model = new CnnModel();
        assertEquals(0, model.getEpochsTrained());
    }
}
