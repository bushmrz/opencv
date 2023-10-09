import java.awt.image.BufferedImage;
import java.awt.Color;
import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;

public class CannyEdgeDetection {

    private int width;
    private int height;
    private int[] grayscaleImage;
    private int[] gaussianBlurImage;
    private int[] sobelImage;
    private int[] nonMaxSuppressedImage;
    private int[] thresholdedImage;

    public CannyEdgeDetection(BufferedImage image) {
        this.width = image.getWidth();
        this.height = image.getHeight();

        grayscaleImage = new int[width * height];
        gaussianBlurImage = new int[width * height];
        sobelImage = new int[width * height];
        nonMaxSuppressedImage = new int[width * height];
        thresholdedImage = new int[width * height];

        processImage(image);
    }

    // Преобразование изображения в оттенки серого
    private void toGrayScale(BufferedImage image) {
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                Color color = new Color(image.getRGB(i, j));
                int red = color.getRed();
                int green = color.getGreen();
                int blue = color.getBlue();
                int gray = (red + green + blue) / 3;
                grayscaleImage[j * width + i] = gray;
            }
        }
    }

    // Применение гауссового размытия
    private void applyGaussianBlur() {
        int[][] kernel = {
                {1, 2, 1},
                {2, 4, 2},
                {1, 2, 1}
        };

        int kernelSize = 3;
        int radius = kernelSize / 2;

        for (int i = radius; i < width - radius; i++) {
            for (int j = radius; j < height - radius; j++) {
                int sum = 0;

                for (int k = -radius; k <= radius; k++) {
                    for (int l = -radius; l <= radius; l++) {
                        int pixel = grayscaleImage[(j + l) * width + (i + k)];
                        sum += pixel * kernel[radius + l][radius + k];
                    }
                }

                gaussianBlurImage[j * width + i] = sum / 16; // Нормализация суммы весов ядра
            }
        }
    }

    // Вычисление градиентов с помощью оператора Собеля
    private void calculateGradients() {
        int[][] sobelXKernel = {
                {-1, 0, 1},
                {-2, 0, 2},
                {-1, 0, 1}
        };

        int[][] sobelYKernel = {
                {-1, -2, -1},
                {0, 0, 0},
                {1, 2, 1}
        };

        int radius = 1;

        for (int i = radius; i < width - radius; i++) {
            for (int j = radius; j < height - radius; j++) {
                int gx = 0;
                int gy = 0;

                for (int k = -radius; k <= radius; k++) {
                    for (int l = -radius; l <= radius; l++) {
                        gx += gaussianBlurImage[(j + l) * width + (i + k)] * sobelXKernel[radius + l][radius + k];
                        gy += gaussianBlurImage[(j + l) * width + (i + k)] * sobelYKernel[radius + l][radius + k];
                    }
                }

                int gradient = (int) Math.sqrt(gx * gx + gy * gy);
//                int direction = (int) Math.toDegrees(Math.atan2(gy, gx));
                sobelImage[j * width + i] = gradient;
            }
        }
    }

    // Подавление немаксимальных значений
    private void suppressNonMax() {
        for (int i = 1; i < width - 1; i++) {
            for (int j = 1; j < height - 1; j++) {
                int gradient = sobelImage[j * width + i];
                int direction = (int) Math.toDegrees(Math.atan2(sobelImage[(j + 1) * width + i] - sobelImage[(j - 1) * width + i],sobelImage[j * width + (i + 1)] - sobelImage[j * width + (i - 1)]));
                direction = (direction + 360) % 360; // Преобразование в положительные углы

                boolean isMax = false;

                if ((0 <= direction && direction < 22.5) || (157.5 <= direction && direction <= 180) || direction <= -157.5 || (-22.5 <= direction && direction < 0)) {
                    isMax = gradient >= sobelImage[j * width + (i - 1)] && gradient >= sobelImage[j * width + (i + 1)];
                } else if ((22.5 <= direction && direction < 67.5) || (-112.5 <= direction && direction < -67.5)) {
                    isMax = gradient >= sobelImage[(j - 1) * width + (i + 1)] && gradient >= sobelImage[(j + 1) * width + (i - 1)];
                } else if ((67.5 <= direction && direction < 112.5) || (-67.5 <= direction && direction < -22.5)) {
                    isMax = gradient >= sobelImage[(j - 1) * width + i] && gradient >= sobelImage[(j + 1) * width + i];
                } else if ((112.5 <= direction && direction < 157.5) || (-157.5 <= direction && direction < -112.5)) {
                    isMax = gradient >= sobelImage[(j - 1) * width + (i - 1)] && gradient >= sobelImage[(j + 1) * width + (i + 1)];
                }

                nonMaxSuppressedImage[j * width + i] = isMax ? gradient : 0;
            }
        }
    }

    // Применение двойного порога
    private void applyDoubleThreshold(int lowThreshold, int highThreshold) {
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int pixel = nonMaxSuppressedImage[j * width + i];

                if (pixel >= highThreshold) {
                    thresholdedImage[j * width + i] = 255; // Сильные границы
                } else if (pixel >= lowThreshold) {
                    thresholdedImage[j * width + i] = 128; // Слабые границы
                } else {
                    thresholdedImage[j * width + i] = 0; // Фон
                }
            }
        }
    }

    // Применение гистерезиса
    private void applyHysteresis(int hysteresisThreshold) {
        for (int i = 1; i < width - 1; i++) {
            for (int j = 1; j < height - 1; j++) {
                if (thresholdedImage[j * width + i] == 128) {
                    boolean hasStrongNeighbor = false;

                    for (int k = -1; k <= 1; k++) {
                        for (int l = -1; l <= 1; l++) {
                            if (thresholdedImage[(j + l) * width + (i + k)] == 255) {
                                hasStrongNeighbor = true;
                                break;
                            }
                        }
                    }

                    if (hasStrongNeighbor) {
                        thresholdedImage[j * width + i] = 255; // Сильная граница
                    } else {
                        thresholdedImage[j * width + i] = 0; // Фон
                    }
                }
            }
        }
    }

    // Обработка изображения по шагам
    private void processImage(BufferedImage image) {
        // Преобразование изображения в оттенки серого
        toGrayScale(image);

        // Применение гауссового размытия
        applyGaussianBlur();

        // Вычисление градиентов с помощью оператора Собеля
        calculateGradients();

        // Подавление немаксимальных значений
        suppressNonMax();

        // Применение двойного порога
        int lowThreshold = 30;
        int highThreshold = 70;
        applyDoubleThreshold(lowThreshold, highThreshold);

        // Применение гистерезиса
                int hysteresisThreshold = 100;
                applyHysteresis(hysteresisThreshold);
            }

            // Получение изображения с границами
            public BufferedImage getEdgesImage() {
                BufferedImage edgesImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < height; j++) {
                        int pixel = thresholdedImage[j * width + i];
                        int rgb = new Color(pixel, pixel, pixel).getRGB();
                        edgesImage.setRGB(i, j, rgb);
                    }
                }

                return edgesImage;
            }
        }

class Main {

    public static void main(String[] args) {
        try {
            BufferedImage inputImage = ImageIO.read(new File("C:\\Users\\ASUS\\IdeaProjects\\gauss\\src\\cat.jpg"));
            CannyEdgeDetection edgeDetection = new CannyEdgeDetection(inputImage);

            // Получение изображения с границами
            BufferedImage edgesImage = edgeDetection.getEdgesImage();

            // Сохранение изображения с границами
            ImageIO.write(edgesImage, "jpg", new File("output.jpg"));

            System.out.println("Границы успешно обнаружены и сохранены в файл output.jpg.");
        } catch (IOException e) {
            System.out.println("Ошибка при загрузке изображения: " + e.getMessage());
        }
    }
}
