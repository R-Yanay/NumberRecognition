package com.handwriting.service;

import com.handwriting.config.ModelConfig;
import com.handwriting.dto.RecognitionRequest;
import com.handwriting.dto.RecognitionResponse;
import com.handwriting.model.CnnModel;
import com.handwriting.model.PredictionResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

/**
 * Service für Ziffernerkennung.
 * Konvertiert Base64-Bilder zu INDArray und ruft das CNN-Modell auf.
 */
@Service
public class RecognitionService {

    private static final Logger log = LoggerFactory.getLogger(RecognitionService.class);

    private final ModelTrainer modelTrainer;

    public RecognitionService(ModelTrainer modelTrainer) {
        this.modelTrainer = modelTrainer;
    }

    /**
     * Ziffer erkennen
     *
     * @param request Base64-encoded 28x28 Grayscale-Bild
     * @return Erkannte Ziffer mit Konfidenz und Wahrscheinlichkeiten
     */
    public RecognitionResponse recognize(RecognitionRequest request) {
        CnnModel model = modelTrainer.getCnnModel();

        if (!model.isLoaded()) {
            ModelTrainer.TrainingStatus status = modelTrainer.getStatus();
            if (status == ModelTrainer.TrainingStatus.TRAINING) {
                throw new IllegalStateException("Modell wird gerade trainiert. Bitte warte, bis das Training abgeschlossen ist.");
            }
            if (status == ModelTrainer.TrainingStatus.ERROR) {
                throw new IllegalStateException("Modell-Training ist fehlgeschlagen. Bitte starte die Anwendung neu.");
            }
            throw new IllegalStateException("Modell nicht geladen");
        }

        // Base64 dekodieren
        byte[] imageBytes = Base64.getDecoder().decode(request.getImage());

        // Zu INDArray konvertieren: [1, 1, 28, 28]
        INDArray image = toIndArray(imageBytes);

        // Inferenz
        long startTime = System.currentTimeMillis();
        PredictionResult result = model.predict(image);
        long duration = System.currentTimeMillis() - startTime;

        log.info("Erkannt: {} (Konfidenz: {}%, Dauer: {}ms)",
                result.getDigit(), String.format("%.2f", result.getConfidence() * 100), duration);

        // Wahrscheinlichkeiten für alle Ziffern
        List<Double> probabilities = new ArrayList<>();
        for (int i = 0; i < ModelConfig.NUM_CLASSES; i++) {
            probabilities.add(result.getProbabilities().getDouble(0, i));
        }

        return new RecognitionResponse(result.getDigit(), result.getConfidence(), probabilities);
    }

    /**
     * Base64-Bild (28x28 Grayscale) zu INDArray konvertieren
     *
     * @param bytes 784 Bytes (28x28 Pixel, 0-255)
     * @return INDArray Shape: [1, 1, 28, 28], normalisiert 0.0-1.0
     */
    private INDArray toIndArray(byte[] bytes) {
        if (bytes.length != ModelConfig.IMAGE_HEIGHT * ModelConfig.IMAGE_WIDTH) {
            throw new IllegalArgumentException(
                    "Bild muss " + (ModelConfig.IMAGE_HEIGHT * ModelConfig.IMAGE_WIDTH) +
                            " Bytes haben (28x28), hat " + bytes.length);
        }

        float[] flat = new float[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            flat[i] = (bytes[i] & 0xFF) / 255.0f;
        }

        INDArray image = Nd4j.create(flat);
        image = image.reshape(1, 1, ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH);
        return image;
    }
}
