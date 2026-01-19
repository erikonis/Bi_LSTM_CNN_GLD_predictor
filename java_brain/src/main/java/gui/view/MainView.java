package gui.view;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
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

/**
 * MainView is the primary JavaFX layout for the application.
 * It assembles the top navigation and three main content panes: Inference,
 * Training and Management. Components are exposed as public fields so the
 * controller can bind event handlers and populate selectors.
 */
public class MainView extends BorderPane {
    // Menu
    public MenuItem itemInfer, itemTrain, itemManagement;

    // Infer Pane Components
    public ComboBox<String> modelSelector, tickerSelector;
    public TextField fieldOpen, fieldHigh, fieldLow, fieldClose, fieldVolume;
    public Button btnPredict;
    public TextField fieldOutput;

    //Management Pane Components
    public ComboBox<String> managementModelSelector, managementTickerSelector;
    public Button btnModelDel;
    public Button btnTickerUpdate;
    
    // Train Pane Components
    public ComboBox<String> trainModelSelector, trainTickerSelector, trainDataSplitSelector;
    public TextField fieldModelName, fieldEpochs, fieldBatch, fieldLR, fieldDropout, fieldHidden, fieldInfo;
    public CheckBox cbEarlyStop, cbAutoFeat;
    public Button btnStartTrain;

    private StackPane contentArea;
    private VBox inferPane, trainPane, managementPane;

    /**
     * Construct the main view, create menu and content panes, and show
     * the inference pane by default.
     */
    public MainView() {
        setTop(createMenuBar());
        
        contentArea = new StackPane();
        createInferPane();
        createManagementPane();
        createTrainPane();
        
        contentArea.getChildren().addAll(inferPane, trainPane, managementPane);
        showInfer(); // Default
        setCenter(contentArea);
    }

    /**
     * Create the top navigation menu containing items to switch views.
     *
     * @return configured MenuBar instance
     */
    private MenuBar createMenuBar() {
        MenuBar menuBar = new MenuBar();
        Menu menu = new Menu("Navigation");
        itemInfer = new MenuItem("Infer");
        itemTrain = new MenuItem("Train");
        itemManagement = new MenuItem("Management");
        menu.getItems().addAll(itemInfer, itemTrain, itemManagement);
        menuBar.getMenus().add(menu);
        return menuBar;
    }

    /**
     * Build and configure the inference pane UI elements (model/ticker
     * selectors, OHLCV inputs, predict button and output field).
     */
    private void createInferPane() {
        inferPane = new VBox(10);
        inferPane.setPadding(new Insets(20));
        inferPane.setAlignment(Pos.CENTER);

        // Title consistent with Management Pane
        Label title = new Label("Market Price Inference");
        title.setStyle("-fx-font-size: 16px; -fx-font-weight: bold;");

        // Model and Ticker Selectors
        HBox selectors = new HBox(10, new Label("Model:"), modelSelector = new ComboBox<>(),
                                      new Label("Ticker:"), tickerSelector = new ComboBox<>());
        selectors.setAlignment(Pos.CENTER);
        modelSelector.setEditable(false);
        tickerSelector.setEditable(false);

        // Input Grid for OHLCV
        GridPane inputs = new GridPane();
        inputs.setHgap(10); inputs.setVgap(10);
        inputs.setAlignment(Pos.CENTER);
        
        inputs.add(new Label("Open:"), 0, 0); inputs.add(fieldOpen = new TextField(), 1, 0);
        inputs.add(new Label("High:"), 2, 0); inputs.add(fieldHigh = new TextField(), 3, 0);
        inputs.add(new Label("Low:"), 0, 1);  inputs.add(fieldLow = new TextField(), 1, 1);
        inputs.add(new Label("Close:"), 2, 1); inputs.add(fieldClose = new TextField(), 3, 1);
        inputs.add(new Label("Volume:"), 0, 2); inputs.add(fieldVolume = new TextField(), 1, 2);

        // Predict Button
        btnPredict = new Button("Predict");
        btnPredict.setStyle("-fx-background-color: #2ecc71; -fx-text-fill: white; -fx-font-weight: bold;");

        // Output Field
        HBox outputBox = new HBox(10, new Label("Output (USD):"), fieldOutput = new TextField());
        fieldOutput.setEditable(false);
        outputBox.setAlignment(Pos.CENTER);

        inferPane.getChildren().addAll(title, selectors, inputs, btnPredict, outputBox);
    }

    /**
     * Build and configure the management pane UI elements used to delete
     * models and update datasets.
     */
    private void createManagementPane() {
    managementPane = new VBox(20);
    managementPane.setPadding(new Insets(20));
    managementPane.setAlignment(Pos.CENTER);

    Label title = new Label("Model & Dataset Management");
    title.setStyle("-fx-font-size: 16px; -fx-font-weight: bold;");

    // Layout for Selectors and their respective buttons
    GridPane managementGrid = new GridPane();
    managementGrid.setHgap(20);
    managementGrid.setVgap(15);
    managementGrid.setAlignment(Pos.CENTER);

    // Column 0: Model Management
    VBox modelBox = new VBox(5);
    modelBox.setAlignment(Pos.CENTER);
    managementModelSelector = new ComboBox<>();
    managementModelSelector.setPromptText("Select Model");
    managementModelSelector.setPrefWidth(150);
    
    btnModelDel = new Button("Delete Model");
    btnModelDel.setStyle("-fx-background-color: #e74c3c; -fx-text-fill: white; -fx-font-weight: bold;");
    btnModelDel.setPrefWidth(150);
    
    modelBox.getChildren().addAll(new Label("Model:"), managementModelSelector, btnModelDel);

    // Column 1: Ticker/Dataset Management
    VBox tickerBox = new VBox(5);
    tickerBox.setAlignment(Pos.CENTER);
    managementTickerSelector = new ComboBox<>();
    managementTickerSelector.setPromptText("Select Ticker");
    managementTickerSelector.setPrefWidth(150);
    
    btnTickerUpdate = new Button("Update Dataset");
    btnTickerUpdate.setStyle("-fx-background-color: #3498db; -fx-text-fill: white; -fx-font-weight: bold;");
    btnTickerUpdate.setPrefWidth(150);
    
    tickerBox.getChildren().addAll(new Label("Ticker:"), managementTickerSelector, btnTickerUpdate);

    // Add boxes to the grid
    managementGrid.add(modelBox, 0, 0);
    managementGrid.add(tickerBox, 1, 0);

    managementPane.getChildren().addAll(title, managementGrid);
}

    /**
     * Build and configure the training pane UI elements for model
     * configuration and training control.
     */
    private void createTrainPane() {
    trainPane = new VBox(15);
    trainPane.setPadding(new Insets(20));
    trainPane.setAlignment(Pos.CENTER);

    Label title = new Label("Deep Learning Training Configuration");
    title.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");

    GridPane grid = new GridPane();
    grid.setHgap(15); grid.setVgap(10);
    grid.setAlignment(Pos.CENTER);

    // Identifiers
    grid.add(new Label("Save Name:"), 0, 0); grid.add(fieldModelName = new TextField("my_model"), 1, 0);
    grid.add(new Label("Architecture:"), 0, 1); grid.add(trainModelSelector = new ComboBox<>(), 1, 1);
    grid.add(new Label("Ticker:"), 0, 2); grid.add(trainTickerSelector = new ComboBox<>(), 1, 2);
    
    // Hyperparameters
    grid.add(new Label("Epochs:"), 2, 0); grid.add(fieldEpochs = new TextField("100"), 3, 0);
    grid.add(new Label("Batch Size:"), 2, 1); grid.add(fieldBatch = new TextField("64"), 3, 1);
    grid.add(new Label("Learning Rate:"), 2, 2); grid.add(fieldLR = new TextField("0.001"), 3, 2);

    // Architecture Details
    grid.add(new Label("Dropout:"), 0, 3); grid.add(fieldDropout = new TextField("0.3"), 1, 3);
    grid.add(new Label("Hidden Dim:"), 2, 3); grid.add(fieldHidden = new TextField("64"), 3, 3);
    
    // Logic Switches
    grid.add(cbEarlyStop = new CheckBox("Early Stopping"), 0, 4);
    grid.add(cbAutoFeat = new CheckBox("Auto Feature Selection"), 1, 4);
    grid.add(new Label("Data Split:"), 2, 4); grid.add(trainDataSplitSelector = new ComboBox<>(), 3, 4);

    grid.add(new Label("Additional Info:"), 0, 5); grid.add(fieldInfo = new TextField("None"), 1, 5, 3, 1);

    btnStartTrain = new Button("Train");
    btnStartTrain.setStyle("-fx-background-color: #d35400; -fx-text-fill: white; -fx-font-weight: bold; -fx-font-size: 14px;");
    btnStartTrain.setPrefWidth(300);

    trainPane.getChildren().addAll(title, grid, btnStartTrain);
}

    /**
     * Create placeholder panes used while parts of the UI are under
     * construction or temporarily disabled.
     */
    private void createPlaceholderPanes() {
        trainPane = new VBox(new Label("Under Construction"));
        trainPane.setAlignment(Pos.CENTER);
    }

    /**
     * Display the inference pane and hide the others.
     */
    public void showInfer() { setVisible(inferPane); }

    /**
     * Display the training pane and hide the others.
     */
    public void showTrain() { setVisible(trainPane); }

    /**
     * Display the management pane and hide the others.
     */
    public void showManagement() { setVisible(managementPane); }

    /**
     * Internal utility to toggle visibility among the content panes.
     *
     * @param target the pane to show (one of inferPane, trainPane,
     *               managementPane)
     */
    private void setVisible(VBox target) {
        inferPane.setVisible(target == inferPane);
        trainPane.setVisible(target == trainPane);
        managementPane.setVisible(target == managementPane);
    }
}