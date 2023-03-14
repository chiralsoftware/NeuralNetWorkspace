package chiralsoftware.netfromscratch;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

/**
 * Read the IDX stream format header, used to hold the MNIST sample images
 */
public record IdxHeader(int dataType, int[] dimensions) {
    
    public static IdxHeader read(InputStream is) throws IOException {
        final byte[] header = new byte[4];
        is.read(header);
        if(! (header[0] == 0 && header[1] == 0)) 
            throw new IOException("invalid magic number - it was " + header[0] + ", " + header[1]);
        final int dt = header[2];
        final int dims = header[3];
        final int[] dimensions = new int[dims];
        final byte[] oneInt = new byte[4];
        for(int i = 0; i < dimensions.length; i++) {
            is.read(oneInt);
            dimensions[i] = oneInt[0] & 0xff << 24 |
                oneInt[1] & 0xff << 16 |
                oneInt[2] & 0xff << 8 |
                oneInt[3] & 0xff;
        }
        return new IdxHeader(dt, dimensions);
    }
    
}
