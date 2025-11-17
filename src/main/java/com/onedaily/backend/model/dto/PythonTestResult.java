package com.onedaily.backend.model.dto;

import lombok.Data;

@Data
public class PythonTestResult {
    private boolean success;
    private String message;
    private String timestamp;
    private String pythonVersion;
    private String error;
}
