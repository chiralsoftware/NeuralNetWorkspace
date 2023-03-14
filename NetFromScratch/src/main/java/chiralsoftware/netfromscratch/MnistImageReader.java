package chiralsoftware.netfromscratch;

import static com.google.common.io.ByteStreams.readFully;
import java.io.FileInputStream;
import java.io.IOException;
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
        
//        final byte[] letsSee = new byte[20];
//        labelGzin.read(letsSee);
//        for(int i = 0; i < letsSee.length; i++) {
//            out.println(i + ": " + letsSee[i]);
//        }
//        
        final IdxHeader imagesIdxHeader = IdxHeader.read(imageGzin);
        final IdxHeader labelIdxHeader = IdxHeader.read(labelGzin);
        
        out.println("Images idx header: " + imagesIdxHeader.dataType() + ", " + Arrays.toString(imagesIdxHeader.dimensions()));
        out.println("Labels idx header: " + labelIdxHeader.dataType() + ", " + Arrays.toString(labelIdxHeader.dimensions()));
        
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
        
//        final Image[] images = new Image[imagesIdxHeader.dimensions()[0]];
        // for some reason this file seems truncated at 60000 images
        final Image[] images = new Image[60000];
        for(int i = 0; i < images.length; i++) {
            final byte[] oneImage = new byte[28*28];
            readFully(imageGzin, oneImage);
            images[i] = new Image(labelGzin.read(), oneImage);
        }
        
        out.println("Read in: " + images.length + " images. here are a few: ");
        out.println(images[0].show());
        out.println("--------------------");
        out.println(images[1].show());
        out.println("--------------------");
        out.println(images[2].show());
        out.println("--------------------");
        out.println(images[3].show());
        out.println("--------------------");
        out.println(images[101].show());
        out.println();
        
    }
    
}
