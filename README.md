# Handschrifterkennung mit Java

Spring Boot Anwendung, die handschriftliche Ziffern (0–9) mit einem Convolutional Neural Network (CNN) erkennt. Das Modell wird in Java mit Deeplearning4j trainiert und über eine REST API deployed. Ein Web-Frontend ermöglicht das Zeichnen mit der Maus.

## Architektur

### Systemdatenfluss

```mermaid
flowchart LR
    subgraph Browser["Browser (Frontend)"]
        Canvas[Canvas 280×280]
        Preproc[Canvas-Bildverarbeitung]
    end

    subgraph Server["Spring Boot (Port 8081)"]
        Controller[RecognitionController]
        Service[RecognitionService]
        Cnn[CnnModel]
        Trainer[ModelTrainer]
        ModelFile[("model.zip")]
    end

    subgraph Training["Training (asynchron)"]
        Mnist[MNIST-Dataset]
        Epochs[3 Epochen]
    end

    Canvas --> Preproc
    Preproc -->|Base64 28×28| Controller
    Controller --> Service
    Service -->|INDArray [1,1,28,28]| Cnn
    Cnn -->|Wahrscheinlichkeiten| Controller
    Controller -->|JSON| Browser

    Trainer -->|lädt| ModelFile
    ModelFile -.->|bei Start| Cnn

    Mnist -->|60k Bilder| Epochs
    Epochs -->|trainiert| Cnn
    Cnn -->|speichert| ModelFile
```

### CNN-Architektur

```mermaid
flowchart LR
    Input["Input\n28×28×1"] --> Conv1["Conv2D(32)\nReLU"]
    Conv1 --> Pool1["MaxPool 2×2"]
    Pool1 --> Conv2["Conv2D(64)\nReLU"]
    Conv2 --> Pool2["MaxPool 2×2"]
    Pool2 --> Flat["Flatten"]
    Flat --> Dense["Dense(128)\nReLU"]
    Dense --> Drop["Dropout(0.5)"]
    Drop --> Output["Output(10)\nSoftmax"]
```

**CNN-Layer im Detail:**

```
Input (28×28) → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(128) → Dropout(0.5) → Output(10, Softmax)
```

## Schnellstart

### Lokal starten

```bash
mvn spring-boot:run
```

Beim ersten Start wird das Modell automatisch mit dem MNIST-Datensatz trainiert (~3-5 Minuten, 3 Epochen). Das Training läuft asynchron im Hintergrund, der Webserver ist sofort verfügbar. Danach wird das trainierte Modell aus `model.zip` geladen.

### Docker

```bash
docker build -t handwriting-recognition .
docker run -p 8081:8081 handwriting-recognition
```

## Web-Frontend

Öffne `http://localhost:8081` im Browser:

1. Zeichne eine Ziffer (0–9) auf dem Canvas
2. Klicke auf "Erkennen"
3. Das CNN erkennt die Ziffer und zeigt Konfidenz + Wahrscheinlichkeiten an

## API-Endpunkte

| Endpunkt | Methode | Beschreibung |
|----------|---------|-------------|
| `/api/recognize` | POST | Ziffer erkennen (Base64-Bild) |
| `/api/model/status` | GET | Trainingsstatus und Modell-Informationen |
| `/api/train` | POST | Training neu starten |

### Beispiel: Erkennung

```bash
curl -X POST http://localhost:8081/api/recognize \
  -H "Content-Type: application/json" \
  -d '{"base64": "BASE64_ENCODED_28x28_IMAGE"}'
```

Antwort:

```json
{
  "digit": 7,
  "confidence": 0.9823,
  "probabilities": [0.0001, 0.0005, 0.0002, 0.0001, 0.0003, 0.0002, 0.0001, 0.9823, 0.0001, 0.0001]
}
```

Die `probabilities`-Liste enthält die Wahrscheinlichkeiten für die Ziffern 0–9 in Reihenfolge (Index 0 = Ziffer 0, Index 1 = Ziffer 1, usw.).

### Beispiel: Modell-Status

```bash
curl http://localhost:8081/api/model/status
```

Antwort:

```json
{
  "ready": true,
  "status": "READY",
  "accuracy": 98.63,
  "epochsTrained": 3,
  "modelLoaded": true
}
```

## CNN-Training

Das Modell wird beim ersten Start automatisch trainiert:

- **Datensatz:** MNIST (60.000 Trainings-, 10.000 Test-Bilder)
- **Epochen:** 3
- **Batch Size:** 256
- **Learning Rate:** 0.001 (Adam-Optimizer)
- **Erreichte Accuracy:** ~98.63%

Das Training läuft asynchron im Hintergrund — der Webserver ist sofort verfügbar. Der Status kann über `/api/model/status` abgefragt werden. Das trainierte Modell wird als `model.zip` gespeichert und beim nächsten Start in wenigen Sekunden geladen.

**Automatische Backups:** Nach jedem Training werden automatisch Backups auf Desktop und Documents erstellt.

## Projektstruktur

### Verzeichnisbaum

```
src/main/java/com/handwriting/
├── HandwritingRecognitionApplication.java  # Spring Boot Entry Point
├── config/
│   └── ModelConfig.java                    # Hyperparameter
├── data/
│   ├── MnistLoader.java                    # MNIST-Datensatz laden
│   └── MnistData.java                      # Datenklasse
├── model/
│   ├── CnnModel.java                       # CNN-Architektur, Training, Inferenz
│   └── PredictionResult.java               # Erkennungsergebnis
├── dto/
│   ├── RecognitionRequest.java             # API-Anfrage
│   └── RecognitionResponse.java            # API-Antwort
├── service/
│   ├── ModelTrainer.java                   # Automatisches Training
│   └── RecognitionService.java             # Bildvorverarbeitung
└── controller/
    └── RecognitionController.java          # REST-Endpunkte
```

### Klassendiagramm

```mermaid
classDiagram
    direction LR

    class HandwritingRecognitionApplication {
        +main(String[] args)
    }

    class ModelConfig {
        <<config>>
        +int EPOCHS
        +int BATCH_SIZE
        +double LEARNING_RATE
    }

    class MnistLoader {
        +loadTrainingData() MnistData
        +loadTestData() MnistData
        -parseImages(File, int) INDArray
        -parseLabels(File, int) INDArray
    }

    class MnistData {
        +INDArray features
        +INDArray labels
        +int getNumSamples()
    }

    class CnnModel {
        +MultiLayerNetwork network
        +double testAccuracy
        +int epochsTrained
        +buildModel()
        +train(MnistData, MnistData)
        +predict(INDArray) double[]
        +load(MnistData)
        +save(File)
        +isLoaded() boolean
    }

    class PredictionResult {
        +int digit
        +double confidence
        +List~Double~ probabilities
    }

    class RecognitionRequest {
        +String image
        +getImage() String
        +setImage(String)
    }

    class RecognitionResponse {
        +int digit
        +double confidence
        +List~Double~ probabilities
    }

    class ModelTrainer {
        +CnnModel cnnModel
        +TrainingStatus status
        +boolean modelLoaded
        +init()
        +doTraining()
        +trainAndSave()
        +getStatus() TrainingStatus
        +createBackup()
    }

    class RecognitionService {
        +ModelTrainer modelTrainer
        +recognize(RecognitionRequest) RecognitionResponse
    }

    class RecognitionController {
        +RecognitionService service
        +ModelTrainer trainer
        +recognize(RecognitionRequest) RecognitionResponse
        +getModelStatus() Map~String,Object~
        +retrain() String
    }

    %% Beziehungen
    HandwritingRecognitionApplication --> RecognitionController : startet
    RecognitionController --> RecognitionService : verwendet
    RecognitionController --> ModelTrainer : fragt Status
    RecognitionService --> ModelTrainer : verwendet
    ModelTrainer --> CnnModel : trainiert und lädt
    ModelTrainer --> MnistLoader : lädt Daten
    ModelTrainer --> ModelConfig : liest Parameter
    CnnModel --> MnistData : trainiert mit
    CnnModel --> PredictionResult : gibt zurück
    RecognitionService --> RecognitionRequest : empfängt
    RecognitionService --> RecognitionResponse : erstellt
    MnistLoader --> MnistData : erstellt
    CnnModel ..> ModelConfig : konfiguriert mit
```

## Technologien

| Technologie | Version | Verwendung |
|-------------|---------|-----------|
| Spring Boot | 3.4.x | REST API, Actuator |
| Deeplearning4j | 1.0.0-M3.1 | CNN-Training und Inferenz |
| ND4J | 1.0.0-M3.1 | Numerische Arrays |
| Java | 25 | Sprache |
| Docker | — | Containerisierung |

## CI/CD

GitHub Actions testet und baut bei jedem Push auf `main`. Die JAR-Datei wird als Artifact gespeichert.

## Lessons Learned

- **CNN vs. simples neuronales Netz:** CNN erreicht ~99% Accuracy, ein simples Feedforward-Netz nur ~97% — Convolution-Layer erkennen Kanten und Muster besser.
- **Dropout ist wichtig:** Ohne Dropout (0.5) überfitet das Modell — Training Accuracy 99.5%, Test Accuracy nur 97%.
- **Mini-Batch Training:** Batch Size 256 bietet gute Geschwindigkeit bei hoher Genauigkeit.
- **One-Hot-Encoding:** Labels als [0,0,0,1,0,0,0,0,0,0] statt als Integer — notwendig für Softmax-Output.
- **Bildnormalisierung:** Pixel-Werte von 0-255 auf 0.0-1.0 skalieren — beschleunigt das Training.
- **Adam-Optimizer:** Deutlich besser als SGD — erreicht höhere Accuracy in weniger Epochen.
- **biasInit(1) vermeiden:** Falsche Bias-Initialisierung blockiert das Training komplett (Accuracy bleibt bei ~10%).
- **Canvas-Bildverarbeitung:** Die Vorverarbeitung der gezeichneten Bilder (Bounding-Box, Zentrierung, Skalierung) hat großen Einfluss auf die Erkennungsqualität. MNIST-Bilder sind zentriert und haben dünne Striche — Canvas-Zeichnungen müssen entsprechend angepasst werden.
- **Eigene Handschrift:** Das Modell erkennt nur Schreibstile, die den MNIST-Trainingsdaten ähneln. Abweichende Handschriften werden schlechter erkannt, obwohl die Modell-Accuracy hoch ist.
- **Asynchrones Training:** Das Training im Hintergrund starten und den Webserver sofort verfügbar machen verbessert die Benutzererfahrung erheblich.
