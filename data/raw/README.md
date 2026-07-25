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

| Veri seti | Resmî kaynak | Beklenen yerel dosya |
|---|---|---|
| Abalone | [UCI Abalone](https://archive.ics.uci.edu/dataset/1/abalone) | `abalone/abalone.data` |
| California Housing | [scikit-learn `fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) | `california_housing/california_housing.csv` |
| Concrete Compressive Strength | [UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) | `concrete/Concrete_Data.xlsx` |
| Wine Quality Red | [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) | `wine_quality/winequality-red.csv` |
| Air Quality NO₂ | [UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) | `air_quality_no2/AirQualityUCI.xlsx` |
| Servo | [UCI Servo](https://archive.ics.uci.edu/dataset/87/servo) | `servo/servo.data` |
| TGSS 2024 | [TGSS resmî sitesi](https://www.tgss.org.tr/) ve [Veri Erişim Onay Formu](https://www.tgss.org.tr/iletisim?type=dataset) | `tgss/TGSS2024.csv` |

UCI Concrete Compressive Strength sayfasındaki `Concrete_Data.xls` dosyası,
Excel veya LibreOffice ile `Concrete_Data.xlsx` adıyla kaydedilmelidir.

California Housing yerel CSV dosyası aşağıdaki komutla oluşturulabilir:

```python
from pathlib import Path
from sklearn.datasets import fetch_california_housing

output = Path("data/raw/california_housing/california_housing.csv")
output.parent.mkdir(parents=True, exist_ok=True)
fetch_california_housing(as_frame=True).frame.to_csv(output, index=False)
```

TGSS 2024 verisi yalnız resmî erişim ve kullanım koşulları altında edinilir.
Erişim onayı, verinin herkese açık biçimde yeniden dağıtılmasına izin verildiği
anlamına gelmez.

## Gizlilik sınırı

Ham TGSS dosyası, hazırlanmış satır düzeyi TGSS verisi ve satır düzeyi tahminler bu depoda yayımlanmaz. Ham veri yolları yalnız çalıştırma betiklerinin beklediği yerel şemayı göstermek amacıyla belgelenmiştir.

Alternatif ham veri kökü `THESIS_RAW_DATA_DIR` ortam değişkeniyle belirtilebilir.
