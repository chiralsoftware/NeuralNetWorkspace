package chiralsoftware.netfromscratch;

import java.io.File;
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
    }
    
}
