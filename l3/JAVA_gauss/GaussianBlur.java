import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class GaussianBlur {
    public static void main(String[] args) {
        try {
            File inputFile = new File("C:\\Users\\ASUS\\IdeaProjects\\gauss\\src\\cat.jpg");
            BufferedImage inputImage = ImageIO.read(inputFile);

            int radius = 5; // радиус размытия
            BufferedImage blurredImage = applyGaussianBlur(inputImage, radius);

            File outputFile = new File("output.jpg");
            ImageIO.write(blurredImage, "jpg", outputFile);

            System.out.println("Размытое изображение сохранено в файл output.jpg");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public static BufferedImage applyGaussianBlur(BufferedImage image, int radius) {
        int width = image.getWidth();
        int height = image.getHeight();

        BufferedImage blurredImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        float[] kernel = createGaussianKernel(radius);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sumRed = 0;
                float sumGreen = 0;
                float sumBlue = 0;
                float sumKernel = 0;

                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int neighborX = x + kx;
                        int neighborY = y + ky;
                        if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
                            int rgb = image.getRGB(neighborX, neighborY);
                            float weight = kernel[ky + radius] * kernel[kx + radius];
                            sumRed += weight * ((rgb >> 16) & 0xFF);
                            sumGreen += weight * ((rgb >> 8) & 0xFF);
                            sumBlue += weight * (rgb & 0xFF);
                            sumKernel += weight;
                        }
                    }
                }

                int pixelRed = Math.round(sumRed / sumKernel);
                int pixelGreen = Math.round(sumGreen / sumKernel);
                int pixelBlue = Math.round(sumBlue / sumKernel);
                int blurredRgb = (pixelRed << 16) | (pixelGreen << 8) | pixelBlue;
                blurredImage.setRGB(x, y, blurredRgb);
            }
        }

        return blurredImage;
    }

    private static float[] createGaussianKernel(int radius) {
        int size = (2 * radius) + 1;
        float[] kernel = new float[size];

        double sigma = radius / 3.0;
        double twoSigmaSquare = 2.0 * sigma * sigma;
        double sigmaRoot = Math.sqrt(twoSigmaSquare * Math.PI);
        double total = 0;

        for (int i = -radius; i <= radius; i++) {
            double distance = i * i;
            int index = i + radius;
            kernel[index] = (float) Math.exp(-distance / twoSigmaSquare) / (float) sigmaRoot;
            total += kernel[index];
        }

        for (int i = 0; i < size; i++) {
            kernel[i] /= total;
        }

        return kernel;
    }
}
