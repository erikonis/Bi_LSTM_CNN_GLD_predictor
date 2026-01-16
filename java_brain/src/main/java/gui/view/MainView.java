package gui.view;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;

public class MainView extends BorderPane {
    // Menu
    public MenuItem itemInfer, itemTrain, itemSettings;

    // Infer Pane Components
    public ComboBox<String> modelSelector, tickerSelector;
    public TextField fieldOpen, fieldHigh, fieldLow, fieldClose, fieldVolume;
    public Button btnPredict;
    public TextField fieldOutput;
    
    private StackPane contentArea;
    private VBox inferPane, trainPane, settingsPane;

    public MainView() {
        setTop(createMenuBar());
        
        contentArea = new StackPane();
        createInferPane();
        createPlaceholderPanes();
        
        contentArea.getChildren().addAll(inferPane, trainPane, settingsPane);
        showInfer(); // Default
        setCenter(contentArea);
    }

    private MenuBar createMenuBar() {
        MenuBar menuBar = new MenuBar();
        Menu menu = new Menu("Navigation");
        itemInfer = new MenuItem("Infer");
        itemTrain = new MenuItem("Train");
        itemSettings = new MenuItem("Settings");
        menu.getItems().addAll(itemInfer, itemTrain, itemSettings);
        menuBar.getMenus().add(menu);
        return menuBar;
    }

    private void createInferPane() {
        inferPane = new VBox(10);
        inferPane.setPadding(new Insets(20));
        inferPane.setAlignment(Pos.CENTER);

        HBox selectors = new HBox(10, new Label("Model:"), modelSelector = new ComboBox<>(),
                                      new Label("Ticker:"), tickerSelector = new ComboBox<>());
        selectors.setAlignment(Pos.CENTER);
        modelSelector.setEditable(false);
        tickerSelector.setEditable(false);

        GridPane inputs = new GridPane();
        inputs.setHgap(10); inputs.setVgap(10);
        inputs.setAlignment(Pos.CENTER);
        
        inputs.add(new Label("Open:"), 0, 0); inputs.add(fieldOpen = new TextField(), 1, 0);
        inputs.add(new Label("High:"), 2, 0); inputs.add(fieldHigh = new TextField(), 3, 0);
        inputs.add(new Label("Low:"), 0, 1);  inputs.add(fieldLow = new TextField(), 1, 1);
        inputs.add(new Label("Close:"), 2, 1); inputs.add(fieldClose = new TextField(), 3, 1);
        inputs.add(new Label("Volume:"), 0, 2); inputs.add(fieldVolume = new TextField(), 1, 2);

        btnPredict = new Button("Predict");
        btnPredict.setStyle("-fx-background-color: #2ecc71; -fx-text-fill: white; -fx-font-weight: bold;");

        HBox outputBox = new HBox(10, new Label("Output (USD):"), fieldOutput = new TextField());
        fieldOutput.setEditable(false);
        outputBox.setAlignment(Pos.CENTER);

        inferPane.getChildren().addAll(selectors, inputs, btnPredict, outputBox);
    }

    private void createPlaceholderPanes() {
        trainPane = new VBox(new Label("Under Construction"));
        trainPane.setAlignment(Pos.CENTER);
        settingsPane = new VBox(new Label("Under Construction"));
        settingsPane.setAlignment(Pos.CENTER);
    }

    public void showInfer() { setVisible(inferPane); }
    public void showTrain() { setVisible(trainPane); }
    public void showSettings() { setVisible(settingsPane); }

    private void setVisible(VBox target) {
        inferPane.setVisible(target == inferPane);
        trainPane.setVisible(target == trainPane);
        settingsPane.setVisible(target == settingsPane);
    }
}