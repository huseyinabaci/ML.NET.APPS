// 1. Veri modeli sınıfı
using Microsoft.ML;
using Microsoft.ML.Data;

public class SalaryData
{
    [LoadColumn(0)] public float YearsExperience { get; set; }
    [LoadColumn(1)] public float EducationLevel { get; set; }
    [LoadColumn(2)] public float Age { get; set; }
    [LoadColumn(3)] public float Salary { get; set; } // Tahmin edilecek değer
}

public class SalaryPrediction
{
    [ColumnName("Score")] public float PredictedSalary { get; set; }
}

class Program
{
    static void Main()
    {
        var context = new MLContext();

        // 2. Veriyi yükle
        string dataPath = ("C:\\Users\\Hüseyin\\source\\repos\\ML.NET.SalaryPredictionApp\\ML.NET.SalaryPredictionApp\\SalaryData.csv");
        if (!File.Exists(dataPath))
        {
            Console.WriteLine("Veri seti bulunamadı! SalaryData.csv dosyasını oluşturduğundan emin ol.");
            return;
        }

        IDataView dataView = context.Data.LoadFromTextFile<SalaryData>(
            dataPath, separatorChar: ',', hasHeader: true);

        // 3. Modeli oluştur
        var pipeline = context.Transforms.Concatenate("Features",
                        nameof(SalaryData.YearsExperience),
                        nameof(SalaryData.EducationLevel),
                        nameof(SalaryData.Age))
                    .Append(context.Regression.Trainers.Sdca(labelColumnName: "Salary", featureColumnName: "Features"));

        // 4. Modeli eğit
        var model = pipeline.Fit(dataView);

        // 5. Kullanıcıdan giriş al
        Console.Write("Deneyim Yılı: ");
        float experience = float.Parse(Console.ReadLine());

        Console.Write("Eğitim Seviyesi (1: Lise, 2: Lisans, 3: Yüksek Lisans): ");
        float education = float.Parse(Console.ReadLine());

        Console.Write("Yaş: ");
        float age = float.Parse(Console.ReadLine());

        // 6. Tahmin yap
        var predictionEngine = context.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(model);
        var sample = new SalaryData { YearsExperience = experience, EducationLevel = education, Age = age };
        var prediction = predictionEngine.Predict(sample);

        Console.WriteLine($"\nTahmini Maaş: {prediction.PredictedSalary:C}");
    }
}