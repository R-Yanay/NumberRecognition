package com.handwriting.service;

import com.handwriting.config.ModelConfig;
import com.handwriting.data.MnistData;
import com.handwriting.data.MnistLoader;
import com.handwriting.model.CnnModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Trainiert das CNN-Modell beim App-Start oder lädt das gespeicherte Modell.
 * Das Training läuft asynchron im Hintergrund, damit der Webserver sofort verfügbar ist.
 */
@Service
public class ModelTrainer {

    private static final Logger log = LoggerFactory.getLogger(ModelTrainer.class);

    private final CnnModel cnnModel = new CnnModel();
    private volatile TrainingStatus status = TrainingStatus.IDLE;

    /**
     * Enum für den Trainingsstatus
     */
    public enum TrainingStatus {
        IDLE,           // Kein Training nötig (Modell geladen)
        TRAINING,       // Training läuft im Hintergrund
        READY,          // Training abgeschlossen, Modell bereit
        ERROR           // Training fehlgeschlagen
    }

    @PostConstruct
    public void init() {
        log.info("Initialisiere CNN-Modell...");

        try {
            MnistLoader loader = new MnistLoader();
            MnistData testData = loader.loadTestData();

            if (cnnModel.load(ModelConfig.MODEL_FILE, testData)) {
                double acc = cnnModel.getTestAccuracy() * 100;
                log.info("Gespeichertes Modell geladen (Accuracy: {}%)", String.format("%.2f", acc));
                this.status = TrainingStatus.READY;
                return;
            }
        } catch (Exception e) {
            log.warn("Modell konnte nicht geladen werden: {}. Starte neues Training.", e.getMessage());
        }

        log.info("Starte Hintergrund-Training...");
        this.status = TrainingStatus.TRAINING;
        new Thread(this::doTraining, "CNN-Training").start();
    }

    /**
     * Führt das Training in einem separaten Thread aus.
     * Der Webserver bleibt sofort verfügbar.
     */
    private void doTraining() {
        try {
            MnistLoader loader = new MnistLoader();

            log.info("Lade MNIST-Trainingsdaten...");
            MnistData trainingData = loader.loadTrainingData();
            log.info("Trainingsdaten geladen: {}", trainingData);

            log.info("Lade MNIST-Testdaten...");
            MnistData testData = loader.loadTestData();
            log.info("Testdaten geladen: {}", testData);

            log.info("Starte CNN-Training...");
            cnnModel.train(trainingData, testData);

            cnnModel.save(ModelConfig.MODEL_FILE);
            double acc = cnnModel.getTestAccuracy() * 100;
            log.info("Modell trainiert und gespeichert. Accuracy: {}%", String.format("%.2f", acc));

            // Automatisch Backup erstellen
            createBackup(ModelConfig.MODEL_FILE);

            this.status = TrainingStatus.READY;
            log.info("Modell bereit für Erkennung!");

        } catch (Exception e) {
            log.error("Fehler beim Training: {}", e.getMessage(), e);
            this.status = TrainingStatus.ERROR;
        }
    }

    /**
     * Erstellt ein Backup der model.zip mit Zeitstempel an mehreren Orten.
     */
    private void createBackup(String modelFile) {
        try {
            File source = new File(modelFile);
            if (!source.exists()) {
                log.warn("Backup übersprungen: model.zip nicht gefunden.");
                return;
            }

            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
            String backupName = "model-backup-" + timestamp + ".zip";

            // Backup 1: Documents Ordner
            Path documents = Paths.get(System.getProperty("user.home"), "Documents");
            Files.createDirectories(documents);
            Path backup1 = documents.resolve(backupName);
            Files.copy(source.toPath(), backup1, StandardCopyOption.REPLACE_EXISTING);
            log.info("Backup 1 erstellt: {}", backup1);

            // Backup 2: Dedizierter Backup-Ordner
            Path backupDir = documents.resolve("HandwritingModelBackups");
            Files.createDirectories(backupDir);
            Path backup2 = backupDir.resolve(backupName);
            Files.copy(source.toPath(), backup2, StandardCopyOption.REPLACE_EXISTING);
            log.info("Backup 2 erstellt: {}", backup2);

            // Backup 3: Desktop
            Path desktop = Paths.get(System.getProperty("user.home"), "Desktop");
            Files.createDirectories(desktop);
            Path backup3 = desktop.resolve(backupName);
            Files.copy(source.toPath(), backup3, StandardCopyOption.REPLACE_EXISTING);
            log.info("Backup 3 (Desktop) erstellt: {}", backup3);

            log.info("Alle Backups erfolgreich erstellt!");

        } catch (IOException e) {
            log.error("Fehler beim Erstellen des Backups: {}", e.getMessage(), e);
        }
    }

    /**
     * MNIST-Daten laden, Modell trainieren und speichern (synchron, für manuellen Aufruf)
     */
    public void trainAndSave() {
        this.status = TrainingStatus.TRAINING;
        try {
            MnistLoader loader = new MnistLoader();

            log.info("Lade MNIST-Trainingsdaten...");
            MnistData trainingData = loader.loadTrainingData();
            log.info("Trainingsdaten geladen: {}", trainingData);

            log.info("Lade MNIST-Testdaten...");
            MnistData testData = loader.loadTestData();
            log.info("Testdaten geladen: {}", testData);

            log.info("Starte CNN-Training...");
            cnnModel.train(trainingData, testData);

            cnnModel.save(ModelConfig.MODEL_FILE);
            double acc2 = cnnModel.getTestAccuracy() * 100;
            log.info("Modell trainiert und gespeichert. Accuracy: {}%", String.format("%.2f", acc2));

            // Automatisch Backup erstellen
            createBackup(ModelConfig.MODEL_FILE);

            this.status = TrainingStatus.READY;

        } catch (Exception e) {
            log.error("Fehler beim Training: {}", e.getMessage(), e);
            this.status = TrainingStatus.ERROR;
            throw new RuntimeException("Training fehlgeschlagen", e);
        }
    }

    public CnnModel getCnnModel() {
        return cnnModel;
    }

    public TrainingStatus getStatus() {
        return status;
    }
}
