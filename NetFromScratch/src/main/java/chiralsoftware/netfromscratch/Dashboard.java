package chiralsoftware.netfromscratch;

import chiralsoftware.netfromscratch.samples.MnistTwoLayer;
import com.googlecode.lanterna.gui2.BasicWindow;
import static com.googlecode.lanterna.gui2.Direction.VERTICAL;
import com.googlecode.lanterna.gui2.GridLayout;
import com.googlecode.lanterna.gui2.Label;
import com.googlecode.lanterna.gui2.LinearLayout;
import com.googlecode.lanterna.gui2.MultiWindowTextGUI;
import com.googlecode.lanterna.gui2.Panel;
import com.googlecode.lanterna.gui2.Window;
import com.googlecode.lanterna.gui2.WindowListenerAdapter;
import com.googlecode.lanterna.gui2.table.Table;
import com.googlecode.lanterna.input.KeyStroke;
import com.googlecode.lanterna.input.KeyType;
import com.googlecode.lanterna.screen.Screen;
import com.googlecode.lanterna.terminal.DefaultTerminalFactory;
import static java.lang.System.Logger.Level.WARNING;
import static java.lang.System.exit;
import static java.lang.System.out;
import java.text.DecimalFormat;
import java.time.Duration;
import static java.time.Duration.between;
import java.time.Instant;
import static java.time.Instant.now;
import java.util.List;
import java.util.concurrent.ExecutorService;
import static java.util.concurrent.Executors.newSingleThreadExecutor;
import static java.util.concurrent.Executors.newSingleThreadScheduledExecutor;
import java.util.concurrent.ScheduledExecutorService;
import static java.util.concurrent.TimeUnit.SECONDS;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 *
 */
public final class Dashboard {

    private static final System.Logger LOG = System.getLogger(Dashboard.class.getName());
    
    private final TrainingTracker trainingTracker = new TrainingTracker();
    
    private static final DefaultTerminalFactory terminalFactory = new DefaultTerminalFactory();

    public static void main(String[] args) throws Exception {
        
        final Dashboard dashboard = new Dashboard();
        // Create terminal and screen
        final Screen screen = terminalFactory.createScreen();
        screen.startScreen();

        // Create GUI manager
        final MultiWindowTextGUI gui = new MultiWindowTextGUI(screen);

        // Create main window
        final BasicWindow window = new BasicWindow("JouleNet Training Dashboard");

        // Panel to hold components
        final Panel root = new Panel(new GridLayout(1));
        final Panel summary = new Panel(new GridLayout(2));
        summary.addComponent( dashboard.epochLabel);
        summary.addComponent(dashboard.lossLabel);
        summary.addComponent(dashboard.timeLabel);
        
        final Panel graphPanel = new Panel();
        graphPanel.addComponent(new Label("Loss graph - todo"));
        
        // Layers
        final Panel layersPanel = new Panel(new LinearLayout(VERTICAL));
        final MnistTwoLayer mnistTwoLayer = new MnistTwoLayer(dashboard.trainingTracker);
        final List<Layer> layers = mnistTwoLayer.getLayers();
        for(int i = 0; i < layers.size(); i++) {
            final Layer layer = layers.get(i);
            dashboard.layersTable.getTableModel().addRow(Integer.toString(i), layer.getClass().getSimpleName(), 
                    Integer.toString(layer.inputSize), Integer.toString(layer.outputSize));
        }
        layersPanel.addComponent(dashboard.layersTable);

        // Controls
        final Label controls = new Label("Press Q to quit | Press P to pause");

        root.addComponent(summary);
        root.addComponent(graphPanel);
        root.addComponent(layersPanel);
        root.addComponent(controls);

        window.setComponent(root);

        window.addWindowListener(new WindowListenerAdapter() {
        @Override
        public void onInput(Window basePane, KeyStroke keyStroke, AtomicBoolean deliverEvent) {
            if (keyStroke.getKeyType() == KeyType.Character && (keyStroke.getCharacter() == 'q' || keyStroke.getCharacter() == 'Q')) {
                exit(0); // not good
                basePane.close();
            }
        }
        }); 
        
        final ScheduledExecutorService executor = newSingleThreadScheduledExecutor();

        // Start periodic GUI updates
        executor.scheduleAtFixedRate(() -> {
            try {
                dashboard.updateGui();
                screen.refresh();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, 0, 1, SECONDS);

        dashboard.startTraining(mnistTwoLayer);
        
        // Show window
        gui.addWindowAndWait(window);

        screen.stopScreen();

    }
    
    private final Label lossLabel = new Label("Loss: -");
    private final Label epochLabel = new Label("Epoch: - / -");
    private final Label timeLabel = new Label("Time: -:--");
    private final DecimalFormat df = new DecimalFormat("0.000000");
    private Instant startTime;

    private final Table<String> layersTable = new Table<>("Layer", "Type", "Inputs", "Outputs");
    
    private void updateGui() {
        lossLabel.setText("Loss: " + df.format(trainingTracker.getAccuracy()));
        epochLabel.setText("Epoch: " + trainingTracker.getEpoch());
        if(startTime!= null) {
            final Duration elapsed = between(startTime, now());
            timeLabel.setText(String.format("Elapsed: %02d:%02d:%02d", 
                    elapsed.toHours(), elapsed.toMinutesPart(), elapsed.toSecondsPart()));
        }
    }
    
    private void startTraining(MnistTwoLayer mnistTwoLayer) throws Exception {
        final ExecutorService trainingExecutor = newSingleThreadExecutor();
        startTime = now();
        trainingExecutor.submit(() -> runTraining(mnistTwoLayer));
    }
    
    private void runTraining(MnistTwoLayer mnistTwoLayer) {
        try { 
            mnistTwoLayer.doIt(); 
        } catch(Exception e) {
            LOG.log(WARNING, "caught", e);
        }
    }
    
}
