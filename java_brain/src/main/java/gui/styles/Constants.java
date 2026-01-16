package gui.styles;

public enum Constants {
    RED_BORDER("-fx-border-color: red; -fx-border-width: 2px; -fx-border-radius: 3px;"),
    DEFAULT_BORDER("-fx-border-color: transparent");

    private final String style;

    Constants(String style) {
        this.style = style;
    }

    @Override
    public String toString() {
        return style;
    }

    public String getStyle() {
        return style;
    }
}