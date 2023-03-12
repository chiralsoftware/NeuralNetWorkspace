package chiralsoftware.netfromscratch;

import java.io.FileInputStream;
import java.util.Arrays;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;

/**
 * Read in the image data and label files for MNIST.
 * Based on code from:
 * https://github.com/ZdsAlpha/MNIST_Extractor/blob/master/MNISTExtractor/Program.cs
 */
public final class MnistImageReader {

    private static final Logger LOG = Logger.getLogger(MnistImageReader.class.getName());
    
    public static void main(String[] args) throws Exception {
        final String trainFileName = "train-images-idx3-ubyte.gz";
        final GZIPInputStream gzin = new GZIPInputStream(new FileInputStream(trainFileName));
        // first 16 bytes are magic number
        final byte[] magicNumber = new byte[16];
        gzin.read(magicNumber);
        System.out.println("Magic number is: " + Arrays.toString(magicNumber));
        
        final byte[] oneImage = new byte[28*28];
        gzin.read(oneImage);
        final int offset = 0;
        final Image image = new Image(1, oneImage);
        System.out.println(image.show());
        System.out.println("I finished reading the image.");
    }
    
}
