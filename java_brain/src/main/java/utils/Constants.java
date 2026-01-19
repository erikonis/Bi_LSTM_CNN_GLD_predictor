package utils;

/**
 * Small enum holding well-known constant names used across the Java
 * application (primarily for logger names).
 */
public enum Constants {
    LOG_COMMS_NAME("COMMS"),
    LOG_BRAIN_NAME("BRAIN");

    private final String value;

    Constants(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    @Override
    public String toString() {
        return getValue();
    }
}
