package chiralsoftware.netfromscratch;

import com.google.common.collect.ImmutableList;
import java.time.Instant;
import static java.time.Instant.now;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.ArrayList;
import java.util.List;

public final class TrainingTracker {
    
    public TrainingTracker(List<Layer> layers) {
        this.layers = layers;
        gradientAverages = new float[layers.size()];
        gradientStandardDeviations = new float[layers.size()];
        gradientSignBalance = new int[layers.size()];
    }
    
    private final List<Layer> layers;
    
    private float accuracy;
    private int parameters;
    private int epoch;
    private Instant startTime = null;
    private final List<Float> lossHistory = new ArrayList<>();
    private final float[] gradientAverages;
    private final float[] gradientStandardDeviations;
    private final int[] gradientSignBalance;
    private String message = null;

    private final ReentrantReadWriteLock rwLock = new ReentrantReadWriteLock();

    public void update(float accuracy, int parameters, int epoch, float loss) {
        rwLock.writeLock().lock();
        try {
            this.accuracy = accuracy;
            this.parameters = parameters;
            this.epoch = epoch;
            lossHistory.add(loss);
        } finally {
            rwLock.writeLock().unlock();
        }
    }
    
    public void reset() {
        rwLock.writeLock().lock();
        try {
            accuracy = 0;
            parameters = 0;
            epoch = 0;
            lossHistory.clear();
        } finally {
            rwLock.writeLock().unlock();
        }
    }

    // Readers: dashboard threads query stats
    public float getAccuracy() {
        rwLock.readLock().lock();
        try {
            return accuracy;
        } finally {
            rwLock.readLock().unlock();
        }
    }

    public int getParameters() {
        rwLock.readLock().lock();
        try {
            return parameters;
        } finally {
            rwLock.readLock().unlock();
        }
    }

    public Instant getStartTime() {
        rwLock.readLock().lock();
        try {
            return startTime;
        } finally {
            rwLock.readLock().unlock();
        }
    }

    public int getEpoch() {
        rwLock.readLock().lock();
        try {
            return epoch;
        } finally {
            rwLock.readLock().unlock();
        }
    }

    public List<Float> getLossHistory() {
        rwLock.readLock().lock();
        try {
            return ImmutableList.copyOf(lossHistory);
        } finally {
            rwLock.readLock().unlock();
        }
    }

    void setStartTime() {
        rwLock.writeLock().lock();
        try {
            startTime = now();
        } finally {
            rwLock.writeLock().unlock();
        }
    }

    void setMessage(String message) {
        rwLock.writeLock().lock();
        try {
            this.message = message;
        } finally {
            rwLock.writeLock().unlock();
        }
    }

    public String getMessage() {
        rwLock.readLock().lock();
        try {
            return message;
        } finally {
            rwLock.readLock().unlock();
        }
    }
    
/**
 * Visualization-only metrics updated each epoch.
 * These arrays are shared between BatchProcessor (writer) and Dashboard (reader).
 * No synchronization is used: occasional read/write overlap may cause minor glitches,
 * which are acceptable for monitoring purposes. Do not persist or use for training logic.
 */ 
    public float[] getGradientAverages() { return gradientAverages; }
    
   /**
 * Visualization-only metrics updated each epoch.
 * These arrays are shared between BatchProcessor (writer) and Dashboard (reader).
 * No synchronization is used: occasional read/write overlap may cause minor glitches,
 * which are acceptable for monitoring purposes. Do not persist or use for training logic.
 */
     public float[] getGradientStrandardDeviations() { return gradientStandardDeviations; }
     
     public int[] getGradientSignBalance() {
         return gradientSignBalance;
     }
}
