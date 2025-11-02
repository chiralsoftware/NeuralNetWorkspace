package chiralsoftware.netfromscratch;

import static com.google.common.io.ByteStreams.readFully;
import java.io.FileInputStream;
import java.io.IOException;
import static java.lang.System.out;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;

/**
 * Read in the image data and label files for MNIST.
 * Based on code from:
 * https://github.com/ZdsAlpha/MNIST_Extractor/blob/master/MNISTExtractor/Program.cs
 */
public final class MnistImageReader {

    private static final Logger LOG = Logger.getLogger(MnistImageReader.class.getName());
    
//        final Image[] images = new Image[imagesIdxHeader.dimensions()[0]];
        // for some reason this file seems truncated at 60000 images
        
    static final Image[] images = new Image[60000];
    
    /** Load the entire MNIST set into memory */
    static void load() throws IOException {
        final String trainFileName = "train-images-idx3-ubyte.gz";
        final String labelFileName = "train-labels-idx1-ubyte.gz";
        final GZIPInputStream imageGzin = new GZIPInputStream(new FileInputStream(trainFileName));
        final GZIPInputStream labelGzin = new GZIPInputStream(new FileInputStream(labelFileName));
        
        final IdxHeader imagesIdxHeader = IdxHeader.read(imageGzin);
        final IdxHeader labelIdxHeader = IdxHeader.read(labelGzin);
        
        if(imagesIdxHeader.dimensions()[1] != 28) {
            throw new IOException("wrong image width; should be 28");
        }
        if(imagesIdxHeader.dimensions()[2] != 28) {
            throw new IOException("wrong image height; should be 28");
        }
        if(imagesIdxHeader.dimensions()[0] != labelIdxHeader.dimensions()[0]) 
            throw new IOException("the number of images and labels don't match");
        
        for(int i = 0; i < images.length; i++) {
            final byte[] oneImage = new byte[28*28];
            readFully(imageGzin, oneImage);
            images[i] = new Image(labelGzin.read(), oneImage);
        }
        
    }

    public static void main(String[] args) throws Exception {
        final String trainFileName = "train-images-idx3-ubyte.gz";
        final String labelFileName = "train-labels-idx1-ubyte.gz";
        final GZIPInputStream imageGzin = new GZIPInputStream(new FileInputStream(trainFileName));
        final GZIPInputStream labelGzin = new GZIPInputStream(new FileInputStream(labelFileName));
        
        final IdxHeader imagesIdxHeader = IdxHeader.read(imageGzin);
        final IdxHeader labelIdxHeader = IdxHeader.read(labelGzin);
        
        if(imagesIdxHeader.dimensions()[1] != 28) {
            out.println("wrong image width; should be 28");
            return;
        }
        if(imagesIdxHeader.dimensions()[2] != 28) {
            out.println("wrong image height; should be 28");
            return;
        }
        if(imagesIdxHeader.dimensions()[0] != labelIdxHeader.dimensions()[0]) {
            out.println("the number of images and labels don't match");
            return;
        }
        
        for(int i = 0; i < images.length; i++) {
            final byte[] oneImage = new byte[28*28];
            readFully(imageGzin, oneImage);
            images[i] = new Image(labelGzin.read(), oneImage);
        }
    
         out.println(images[101].show());
        
    }
    
}
