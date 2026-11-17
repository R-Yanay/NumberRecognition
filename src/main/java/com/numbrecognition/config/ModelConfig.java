package com.numbrecognition.config;

/**
 * Hyperparameter für das CNN-Modell
 */
public class ModelConfig {

    public static final int IMAGE_HEIGHT = 28;
    public static final int IMAGE_WIDTH = 28;
    public static final int NUM_CLASSES = 10;

    // Training
    public static final int EPOCHS = 3;
    public static final int BATCH_SIZE = 256;
    public static final double LEARNING_RATE = 1e-3;

    // CNN-Architektur
    public static final int CONV1_FILTERS = 32;
    public static final int CONV2_FILTERS = 64;
    public static final int DENSE_UNITS = 128;
    public static final double DROPOUT_RATE = 0.5;

    // Modell-Datei
    public static final String MODEL_FILE = "model.zip";
}
