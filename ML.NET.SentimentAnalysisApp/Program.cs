using Microsoft.ML;
using Microsoft.ML.Data;

// 📌 Veri Modeli
public class SentimentData
{
    [LoadColumn(0)] public string Text { get; set; }
    [LoadColumn(1)] public bool Sentiment { get; set; } // 1: Olumlu, 0: Olumsuz
}

// 📌 Tahmin Sonucu Modeli
public class SentimentPrediction
{
    [ColumnName("PredictedLabel")] public bool PredictedLabel { get; set; }
}

class Program
{
    static void Main()
    {
        // ML.NET Ortamını Başlat
        var context = new MLContext();

        // Veri dosyasının yolu
        string dataPath = "C:\\Users\\Hüseyin\\source\\repos\\ML.NET.SalaryPredictionApp\\ML.NET.SentimentAnalysisApp\\SentimentData.csv";

        // Veri Kümesini Yükle
        IDataView dataView = context.Data.LoadFromTextFile<SentimentData>(
            dataPath, separatorChar: ',', hasHeader: true);

        // Veri İşleme ve Model Pipeline Tanımlama
        var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
                      .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Sentiment", featureColumnName: "Features"));

        // Modeli Eğit
        var model = pipeline.Fit(dataView);

        // Tahmin Motorunu Oluştur
        var predictionEngine = context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

        while (true)
        {
            // Kullanıcıdan Giriş Al
            Console.Write("\nBir yorum gir (çıkmak için 'exit' yaz): ");
            string inputText = Console.ReadLine();
            if (inputText.ToLower() == "exit") break;

            // Model ile Tahmin Yap
            var sample = new SentimentData { Text = inputText };
            var prediction = predictionEngine.Predict(sample);

            // Sonucu Ekrana Yazdır
            Console.WriteLine($"Tahmin: {(prediction.PredictedLabel ? "Olumlu 😊" : "Olumsuz 😡")}");
        }
    }
}
