package chiralsoftware.netfromscratch;

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
import java.io.IOException;
import static java.lang.System.Logger.Level.INFO;
import static java.lang.System.exit;
import java.text.DecimalFormat;
import java.time.Duration;
import static java.time.Duration.between;
import java.time.Instant;
import static java.time.Instant.now;
import java.util.ArrayList;
import static java.util.Arrays.asList;
import static java.util.Collections.shuffle;
import java.util.List;
import java.util.Random;
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
    
    private TrainingTracker trainingTracker;
    
    private final DefaultTerminalFactory terminalFactory = new DefaultTerminalFactory();
    private static final int trainingSampleSize = 1000;
    private final Screen screen;
    private final MultiWindowTextGUI gui;
    private final BasicWindow window; 
    
    private static final ExecutorService trainingExecutor = newSingleThreadExecutor();
    
    private static final ExecutorService windowExecutor = newSingleThreadExecutor();
    
    public static void main(String[] args) throws Exception {
        final ArrayList<Layer> layers = new ArrayList<>();
        final SigmoidActivationLayer sal1 = new SigmoidActivationLayer(28*28, 64);
        sal1.initRandom();
        layers.add(sal1);
        final SigmoidActivationLayer sal2 = new SigmoidActivationLayer(64,32);
        sal2.initRandom();
        layers.add(sal2);
        final SoftMaxLayer soft = new SoftMaxLayer(32, 10);
        soft.initRandom();
        layers.add(soft);

        final Network network = new Network(layers);
        
        LOG.log(INFO, "loading images and preparing training sets");
        final ArrayList<Image> allImages = new ArrayList<>();
        MnistImageReader.load();
        allImages.addAll(asList(MnistImageReader.images));
        shuffle(allImages);
        
        if(trainingSampleSize > allImages.size()) 
            throw new IllegalStateException("training sample size: " + trainingSampleSize + 
                    " was larger than allImages size: " + allImages.size());
        final ArrayList<Sample> samples = new ArrayList<>();
        for(int i = 0; i < trainingSampleSize; i++) {
            final Image image = allImages.get(i);
            if(image == null) throw new NullPointerException("image at: " + i + " was null");
            final float[] imageFloat = new float[28*28];
            image.toFloat(imageFloat);
            final float[] target = new float[10];
            target[image.label()] = 1f;
            samples.add(new Sample<>(imageFloat, target, image));
        }

        final TrainingRun trainingRun = new TrainingRun(samples, network);
        final TrainingTracker trainingTracker = new TrainingTracker(layers);
        network.setTrainingTracker(trainingTracker);
        trainingRun.setTrainingTracker(trainingTracker);
        
        final Dashboard dashboard = new Dashboard(layers);
        dashboard.trainingTracker = trainingTracker;
        startTraining(trainingRun);
        dashboard.startUpdateTask();
        dashboard.start();
    }

    private static void startTraining(TrainingRun trainingRun) throws Exception {
        trainingExecutor.submit(() -> {
            trainingRun.start();
        });
    }
    
    
    Dashboard(List<Layer> layers) throws Exception {
        // Create terminal and screen
        screen = terminalFactory.createScreen();
        screen.startScreen();

        // Create GUI manager
        gui = new MultiWindowTextGUI(screen);

        // Create main window
        window = new BasicWindow("JouleNet Training Dashboard");

        // Panel to hold components
        final Panel root = new Panel(new GridLayout(1));
        final Panel summary = new Panel(new GridLayout(2));
        summary.addComponent(epochLabel);
        summary.addComponent(lossLabel);
        summary.addComponent(timeLabel);
        
        final Panel graphPanel = new Panel();
        graphPanel.addComponent(new Label("Loss graph - todo"));
        
        // Layers
        final Panel layersPanel = new Panel(new LinearLayout(VERTICAL));
        for(int i = 0; i < layers.size(); i++) {
            final Layer layer = layers.get(i);
            layersTable.getTableModel().addRow(Integer.toString(i), layer.getClass().getSimpleName(), 
                    Integer.toString(layer.inputSize), Integer.toString(layer.outputSize), 
                    "-", "-", "-");
        }
        layersPanel.addComponent(layersTable);

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
        
    }
    
    /** Start the executor to update the GUI once a second */
    void start() throws Exception {
        // Show window
        gui.addWindowAndWait(window);
        screen.stopScreen();
    }
    
    void startUpdateTask() {
        final ScheduledExecutorService executor = newSingleThreadScheduledExecutor();

        // Start periodic GUI updates
        executor.scheduleAtFixedRate(() -> {
            try {
                updateGui();
                screen.refresh();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, 0, 1, SECONDS);
        
    }
    
    private final Label lossLabel = new Label("Loss: -");
    private final Label epochLabel = new Label("Epoch: - / -");
    private final Label timeLabel = new Label("Time: -:--");
    private final DecimalFormat df = new DecimalFormat("0.000000");

    private final Table<String> layersTable = new Table<>("Layer", "Type", "Inputs", 
            "Outputs", "Gradient Avg", "Grad. Std Dev", "Grad. Sign Balance");
    
    private void updateGui() throws IOException {
        lossLabel.setText("Loss: " + df.format(trainingTracker.getAccuracy()));
        epochLabel.setText("Epoch: " + trainingTracker.getEpoch());
        final Instant startTime = trainingTracker.getStartTime();
        if(startTime!= null) {
            final Duration elapsed = between(startTime, now());
            timeLabel.setText(String.format("Elapsed: %02d:%02d:%02d", 
                    elapsed.toHours(), elapsed.toMinutesPart(), elapsed.toSecondsPart()));
        }
        for(int l = 0; l < trainingTracker.getGradientAverages().length; l++) {
            layersTable.getTableModel().
                    setCell(4, l, df.format(trainingTracker.getGradientAverages()[l]));
            layersTable.getTableModel().
                    setCell(5, l, df.format(trainingTracker.getGradientStrandardDeviations()[l]));
            layersTable.getTableModel().
                    setCell(6, l, Integer.toString(trainingTracker.getGradientSignBalance()[l]));
        }
        gui.updateScreen();
    }
    
}
