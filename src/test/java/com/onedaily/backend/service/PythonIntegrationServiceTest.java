package com.onedaily.backend.service;

import com.onedaily.backend.model.dto.PythonTestResult;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
class PythonIntegrationServiceTest {

    @Autowired
    private PythonIntegrationService service;

    @Test
    void testPythonConnection() {
        // given
        String testMessage = "Hello Java!";

        // when
        PythonTestResult result = service.testConnection(testMessage);

        // then
        assertThat(result.isSuccess()).isTrue();
        assertThat(result.getMessage()).contains("Hello from Python");
        assertThat(result.getMessage()).contains(testMessage);
        assertThat(result.getPythonVersion()).isNotNull();
    }
}
