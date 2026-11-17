package com.handwriting.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Anfrage für Ziffernerkennung
 */
public class RecognitionRequest {

    @JsonProperty("base64")
    private String image;  // Base64-encoded 28x28 Grayscale-Bild

    public RecognitionRequest() {
    }

    public RecognitionRequest(String image) {
        this.image = image;
    }

    public String getImage() {
        return image;
    }

    public void setImage(String image) {
        this.image = image;
    }
}
