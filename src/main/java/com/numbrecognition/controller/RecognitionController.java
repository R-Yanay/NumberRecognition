package com.numbrecognition.controller;

import com.numbrecognition.dto.RecognitionRequest;
import com.numbrecognition.dto.RecognitionResponse;
import com.numbrecognition.model.CnnModel;
import com.numbrecognition.service.ModelTrainer;
import com.numbrecognition.service.RecognitionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * REST-API für Handschrifterkennung
 */
@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class RecognitionController {

    private static final Logger log = LoggerFactory.getLogger(RecognitionController.class);

    private final RecognitionService recognitionService;
    private final ModelTrainer modelTrainer;

    public RecognitionController(RecognitionService recognitionService, ModelTrainer modelTrainer) {
        this.recognitionService = recognitionService;
        this.modelTrainer = modelTrainer;
    }

    /**
     * POST /api/recognize
     * Erkennt eine handschriftliche Ziffer (0-9)
     *
     * @param request Base64-encoded 28x28 Grayscale-Bild
     * @return Erkannte Ziffer mit Konfidenz und Wahrscheinlichkeiten
     */
    @PostMapping("/recognize")
    public ResponseEntity<RecognitionResponse> recognize(@RequestBody RecognitionRequest request) {
        log.info("Erkennungsanfrage empfangen");

        RecognitionResponse response = recognitionService.recognize(request);

        return ResponseEntity.ok(response);
    }

    /**
     * GET /api/model/status
     * Gibt den aktuellen Trainingsstatus zurück
     *
     * @return Status (IDLE, TRAINING, READY, ERROR)
     */
    @GetMapping("/model/status")
    public ResponseEntity<Map<String, Object>> modelStatus() {
        ModelTrainer.TrainingStatus status = modelTrainer.getStatus();
        CnnModel model = modelTrainer.getCnnModel();

        Map<String, Object> info = Map.of(
                "status", status.name(),
                "ready", status == ModelTrainer.TrainingStatus.READY,
                "loaded", model.isLoaded(),
                "testAccuracy", model.isLoaded() ? String.format("%.2f%%", model.getTestAccuracy() * 100) : "N/A",
                "epochsTrained", model.getEpochsTrained()
        );

        return ResponseEntity.ok(info);
    }

    /**
     * POST /api/train
     * Startet das Training neu (optional)
     *
     * @return Status-Nachricht
     */
    @PostMapping("/train")
    public ResponseEntity<Map<String, String>> train() {
        log.info("Neues Training gestartet (manuell)");

        try {
            modelTrainer.trainAndSave();
            return ResponseEntity.ok(Map.of("status", "Training abgeschlossen",
                    "accuracy", String.format("%.2f%%", modelTrainer.getCnnModel().getTestAccuracy() * 100)));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(Map.of("error", e.getMessage()));
        }
    }

}
