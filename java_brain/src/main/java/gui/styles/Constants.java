package gui.styles;

/**
 * Style constants used by the JavaFX UI for quick inline styling values.
 */
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

    /**
     * Return the CSS style string for this constant.
     *
     * @return CSS fragment
     */
    public String getStyle() {
        return style;
    }
}