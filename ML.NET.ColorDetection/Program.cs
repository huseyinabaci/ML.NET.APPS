using System.Drawing;

class Program
{
    static void Main()
    {
        string imagePath = ("C:\\Users\\Hüseyin\\source\\repos\\ML.NET.SalaryPredictionApp\\ML.NET.ColorDetection\\sample.jpg");
        if (!System.IO.File.Exists(imagePath))
        {
            Console.WriteLine("Resim dosyası bulunamadı!");
            return;
        }

        Console.WriteLine("Renkler Algılanıyor...");
        Dictionary<Color, int> colorFrequency = AnalyzeImageColors(imagePath);

        Console.WriteLine("\nEn Yaygın 5 Renk:");
        foreach (var color in colorFrequency.OrderByDescending(c => c.Value).Take(5))
        {
            string colorName = GetColorName(color.Key);
            Console.WriteLine($"Renk: {colorName} - RGB({color.Key.R}, {color.Key.G}, {color.Key.B}) - {color.Value} Piksel");
        }
    }

    static Dictionary<Color, int> AnalyzeImageColors(string imagePath)
    {
        Bitmap bitmap = new Bitmap(imagePath);
        Dictionary<Color, int> colorCount = new Dictionary<Color, int>();

        for (int x = 0; x < bitmap.Width; x++)
        {
            for (int y = 0; y < bitmap.Height; y++)
            {
                Color pixelColor = bitmap.GetPixel(x, y);

                if (colorCount.ContainsKey(pixelColor))
                    colorCount[pixelColor]++;
                else
                    colorCount[pixelColor] = 1;
            }
        }

        return colorCount;
    }

    static string GetColorName(Color color)
    {
        // Yaygın renkler ve RGB değerleri
        Dictionary<string, Color> knownColors = new Dictionary<string, Color>
        {
            { "Black", Color.Black },
            { "White", Color.White },
            { "Red", Color.Red },
            { "Green", Color.Green },
            { "Blue", Color.Blue },
            { "Yellow", Color.Yellow },
            { "Orange", Color.Orange },
            { "Purple", Color.Purple },
            { "Pink", Color.Pink },
            { "Brown", Color.Brown },
            { "Gray", Color.Gray }
        };

        string closestColor = "Unknown";
        double minDistance = double.MaxValue;

        foreach (var kvp in knownColors)
        {
            double distance = Math.Sqrt(Math.Pow(color.R - kvp.Value.R, 2) +
                                        Math.Pow(color.G - kvp.Value.G, 2) +
                                        Math.Pow(color.B - kvp.Value.B, 2));

            if (distance < minDistance)
            {
                minDistance = distance;
                closestColor = kvp.Key;
            }
        }

        return closestColor;
    }
}