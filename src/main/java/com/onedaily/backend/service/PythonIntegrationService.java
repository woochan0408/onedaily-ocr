package com.onedaily.backend.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.onedaily.backend.model.dto.PythonTestResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;

@Slf4j
@Service
public class PythonIntegrationService {

    private final ObjectMapper objectMapper = new ObjectMapper();

    public PythonTestResult testConnection(String testMessage) {
        try {
            log.info("Testing Python connection with message: {}", testMessage);

            // 1. Python 스크립트 실행
            ProcessBuilder processBuilder = new ProcessBuilder(
                "python3",
                "python-scripts/test_connection.py",
                testMessage
            );

            // 2. 프로세스 시작
            Process process = processBuilder.start();

            // 3. 출력 읽기
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream())
            );
            String jsonOutput = reader.readLine();

            log.info("Python output: {}", jsonOutput);

            // 4. 프로세스 종료 대기
            int exitCode = process.waitFor();
            log.info("Process exit code: {}", exitCode);

            // 5. JSON 파싱
            PythonTestResult result = objectMapper.readValue(
                jsonOutput,
                PythonTestResult.class
            );

            return result;

        } catch (Exception e) {
            log.error("Python integration test failed", e);
            throw new RuntimeException("Python integration failed", e);
        }
    }
}
