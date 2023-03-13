package chiralsoftware.netfromscratch;

import java.io.FileInputStream;
import static java.lang.System.out;
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
        final String labelFileName = "train-labels-idx1-ubyte.gz";
        final GZIPInputStream imageGzin = new GZIPInputStream(new FileInputStream(trainFileName));
        
        final GZIPInputStream labelGzin = new GZIPInputStream(new FileInputStream(labelFileName));
        
        final byte[] letsSee = new byte[200];
        labelGzin.read(letsSee);
        for(int i = 0; i < letsSee.length; i++) {
            out.println(i + ": " + letsSee[i]);
        }
        if(true) return;
        // first 16 bytes are magic number
        final byte[] magicNumber = new byte[16];
        imageGzin.read(magicNumber);
        // see:
        // https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
        // for details about these values
        if(! Arrays.equals(magicNumber, new byte[] {
            0, 0, // always zero 
            8, // values are unsigned bytes
            3, 
            0, 0, -22, 96, 
            0, 0, 0, 28, 0, 0, 0, 28 // this is the shape of the images
        })) {
            System.out.println("file isn't a valid MNIST image file");
            return;
        }
        
        final byte[] oneImage = new byte[28*28];
        imageGzin.read(oneImage);
        final Image image = new Image(1, oneImage);
        System.out.println(image.show());
        System.out.println("I finished reading the image.");
    }
    
}
