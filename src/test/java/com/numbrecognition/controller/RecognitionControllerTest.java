package com.numbrecognition.controller;

import com.numbrecognition.config.ModelConfig;
import com.numbrecognition.dto.RecognitionRequest;
import com.numbrecognition.dto.RecognitionResponse;
import com.numbrecognition.model.CnnModel;
import com.numbrecognition.service.ModelTrainer;
import com.numbrecognition.service.RecognitionService;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Base64;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests für den RecognitionController.
 * Da Java 25 Mockito Inline-Mocking blockiert, werden hier echte Objekte verwendet.
 */
class RecognitionControllerTest {

    private RecognitionController controller;
    private CnnModel cnnModel;
    private ModelTrainer modelTrainer;

    @BeforeEach
    void setUp() throws Exception {
        // Erstelle ein echtes, initialisiertes CNN-Modell
        cnnModel = new CnnModel();
        MultiLayerNetwork network = new MultiLayerNetwork(cnnModel.buildConfiguration());
        network.init();

        java.lang.reflect.Field field = CnnModel.class.getDeclaredField("network");
        field.setAccessible(true);
        field.set(cnnModel, network);

        // ModelTrainer via Reflection konfigurieren
        modelTrainer = new ModelTrainer();
        java.lang.reflect.Field trainerField = ModelTrainer.class.getDeclaredField("cnnModel");
        trainerField.setAccessible(true);
        trainerField.set(modelTrainer, cnnModel);

        java.lang.reflect.Field statusField = ModelTrainer.class.getDeclaredField("status");
        statusField.setAccessible(true);
        statusField.set(modelTrainer, ModelTrainer.TrainingStatus.READY);

        RecognitionService recognitionService = new RecognitionService(modelTrainer);
        controller = new RecognitionController(recognitionService, modelTrainer);
    }

    @Test
    void recognize_returnsOk_withValidInput() {
        byte[] image = new byte[28 * 28];
        for (int i = 0; i < image.length; i++) {
            image[i] = (byte) 200;
        }
        RecognitionRequest request = new RecognitionRequest(Base64.getEncoder().encodeToString(image));

        var responseEntity = controller.recognize(request);

        assertEquals(200, responseEntity.getStatusCodeValue());
        RecognitionResponse body = responseEntity.getBody();
        assertNotNull(body);
        assertTrue(body.getDigit() >= 0 && body.getDigit() <= 9);
        assertTrue(body.getConfidence() > 0);
    }

    @Test
    void modelStatus_returnsReady() throws Exception {
        java.lang.reflect.Field accField = CnnModel.class.getDeclaredField("testAccuracy");
        accField.setAccessible(true);
        accField.setDouble(cnnModel, 0.9863);

        var responseEntity = controller.modelStatus();

        assertEquals(200, responseEntity.getStatusCodeValue());
        Map<String, Object> body = responseEntity.getBody();
        assertNotNull(body);
        assertEquals(true, body.get("ready"));
        assertEquals(true, body.get("loaded"));
        assertEquals("READY", body.get("status"));
    }

    @Test
    void modelStatus_containsAccuracy() throws Exception {
        java.lang.reflect.Field accField = CnnModel.class.getDeclaredField("testAccuracy");
        accField.setAccessible(true);
        accField.setDouble(cnnModel, 0.95);

        java.lang.reflect.Field epochField = CnnModel.class.getDeclaredField("epochsTrained");
        epochField.setAccessible(true);
        epochField.setInt(cnnModel, 3);

        var responseEntity = controller.modelStatus();
        Map<String, Object> body = responseEntity.getBody();

        assertNotNull(body.get("testAccuracy"));
        assertEquals(3, body.get("epochsTrained"));
    }

    @Test
    void recognize_returnsProbabilities() {
        byte[] image = new byte[28 * 28];
        for (int i = 0; i < image.length; i++) {
            image[i] = (byte) 200;
        }
        RecognitionRequest request = new RecognitionRequest(Base64.getEncoder().encodeToString(image));

        var responseEntity = controller.recognize(request);
        RecognitionResponse body = responseEntity.getBody();

        List<Double> probabilities = body.getProbabilities();
        assertEquals(ModelConfig.NUM_CLASSES, probabilities.size());

        // Softmax-Wahrscheinlichkeiten summieren sich zu ~1.0
        double sum = 0;
        for (double p : probabilities) {
            sum += p;
        }
        assertEquals(1.0, sum, 0.01);
    }
}
