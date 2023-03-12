package chiralsoftware.netfromscratch;

import java.io.File;
import static java.lang.System.out;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Logger;

/**
 * Read in the image data and label files for MNIST.
 * Based on code from:
 * https://github.com/ZdsAlpha/MNIST_Extractor/blob/master/MNISTExtractor/Program.cs
 */
public final class MnistImageReader {

    private static final Logger LOG = Logger.getLogger(MnistImageReader.class.getName());
    
    public static void main(String[] args) throws Exception {
        final String fileName = "train-images-idx3-ubyte";
        final File f = new File(fileName);
        if(! f.canRead()) {
            LOG.info("can't read: "+ f);
            return;
        }
        final Path p = f.toPath();
        final byte[] imageFile = Files.readAllBytes(p);
        LOG.info("Read in: "  + imageFile.length  + " bytes");
        LOG.info("First bytes: " + (int) imageFile[0] + ", " + (int) imageFile[1]);
        final int offset = 16 + 1*28*28;
        for(int y  = 0; y < 28; y++) {
            final StringBuilder line = new StringBuilder();
            for(int x = 0; x < 28; x++) {
                out.format("%3d ", imageFile[y * 28 + x + offset] & 0xff);
            }
            out.println();
        }
    }
    
}
