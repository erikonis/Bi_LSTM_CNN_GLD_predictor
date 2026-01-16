package utils;

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
