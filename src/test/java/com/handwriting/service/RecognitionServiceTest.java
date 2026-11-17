package com.handwriting.service;

import com.handwriting.config.ModelConfig;
import com.handwriting.dto.RecognitionRequest;
import com.handwriting.dto.RecognitionResponse;
import com.handwriting.model.CnnModel;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Base64;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests für den RecognitionService.
 * Verwendet ein echtes CnnModel statt Mockito, da Java 25 Mockito Inline-Mocking blockiert.
 */
class RecognitionServiceTest {

    private RecognitionService recognitionService;
    private CnnModel cnnModel;

    private byte[] createTestImage() {
        byte[] image = new byte[28 * 28];
        for (int i = 0; i < image.length; i++) {
            image[i] = (byte) 200;
        }
        return image;
    }

    private String encodeImage(byte[] image) {
        return Base64.getEncoder().encodeToString(image);
    }

    @BeforeEach
    void setUp() throws Exception {
        // Erstelle ein echtes, initialisiertes CNN-Modell
        cnnModel = new CnnModel();
        MultiLayerNetwork network = new MultiLayerNetwork(cnnModel.buildConfiguration());
        network.init();

        // Setze network via Reflection
        java.lang.reflect.Field field = CnnModel.class.getDeclaredField("network");
        field.setAccessible(true);
        field.set(cnnModel, network);

        // ModelTrainer via Reflection mit echtem CnnModel injizieren
        ModelTrainer modelTrainer = new ModelTrainer();
        java.lang.reflect.Field trainerField = ModelTrainer.class.getDeclaredField("cnnModel");
        trainerField.setAccessible(true);
        trainerField.set(modelTrainer, cnnModel);

        java.lang.reflect.Field statusField = ModelTrainer.class.getDeclaredField("status");
        statusField.setAccessible(true);
        statusField.set(modelTrainer, ModelTrainer.TrainingStatus.READY);

        recognitionService = new RecognitionService(modelTrainer);
    }

    @Test
    void recognize_validBase64_returnsResult() {
        byte[] image = createTestImage();
        RecognitionRequest request = new RecognitionRequest(encodeImage(image));

        RecognitionResponse response = recognitionService.recognize(request);

        assertNotNull(response);
        assertTrue(response.getDigit() >= 0 && response.getDigit() <= 9);
        assertTrue(response.getConfidence() > 0);
        assertEquals(ModelConfig.NUM_CLASSES, response.getProbabilities().size());
    }

    @Test
    void recognize_emptyBase64_throwsException() {
        RecognitionRequest request = new RecognitionRequest("");

        assertThrows(Exception.class, () -> recognitionService.recognize(request));
    }

    @Test
    void recognize_wrongSize_throwsException() {
        byte[] wrongImage = new byte[100];
        String base64 = Base64.getEncoder().encodeToString(wrongImage);
        RecognitionRequest request = new RecognitionRequest(base64);

        assertThrows(IllegalArgumentException.class, () -> recognitionService.recognize(request));
    }

    @Test
    void recognize_returnsAllProbabilities() {
        byte[] image = createTestImage();
        RecognitionRequest request = new RecognitionRequest(encodeImage(image));

        RecognitionResponse response = recognitionService.recognize(request);

        List<Double> probabilities = response.getProbabilities();
        assertEquals(10, probabilities.size());

        // Wahrscheinlichkeiten müssen sich zu ~1.0 summieren (Softmax)
        double sum = 0;
        for (double p : probabilities) {
            sum += p;
        }
        assertEquals(1.0, sum, 0.01, "Softmax-Wahrscheinlichkeiten summieren sich zu 1.0");
    }
}
