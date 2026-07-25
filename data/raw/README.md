# Yerel ham veri alanı

Bu dizin yalnız yerel kullanım içindir ve `.gitignore` tarafından izleme dışı bırakılır.

Beklenen yapı:

```text
data/raw/
├── abalone/abalone.data
├── california_housing/california_housing.csv
├── concrete/Concrete_Data.xlsx
├── wine_quality/winequality-red.csv
├── air_quality_no2/AirQualityUCI.xlsx
├── servo/servo.data
└── tgss/TGSS2024.csv
```

## Kaynak notları

- Abalone, Concrete Compressive Strength, Wine Quality Red, Air Quality ve Servo özgün UCI kaynaklarından edinilir.
- California Housing dosyası scikit-learn `fetch_california_housing` ile yerel CSV'ye dönüştürülebilir.
- TGSS 2024 verisi yalnız resmî erişim ve kullanım koşulları altında edinilir.

## Gizlilik sınırı

Ham TGSS dosyası, hazırlanmış satır düzeyi TGSS verisi ve satır düzeyi tahminler bu depoda yayımlanmaz. Ham veri yolları yalnız çalıştırma betiklerinin beklediği yerel şemayı göstermek amacıyla belgelenmiştir.

Alternatif ham veri kökü `THESIS_RAW_DATA_DIR` ortam değişkeniyle belirtilebilir.
