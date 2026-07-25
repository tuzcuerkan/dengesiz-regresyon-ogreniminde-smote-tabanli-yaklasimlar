# Veri erişimi ve gizlilik

## Temel ilke

Bu depo, tezin bilimsel iş akışını ve yeniden üretilebilirlik sınırlarını belgeler. Üçüncü taraf ham verilerin yeniden dağıtım deposu değildir. Ham veriler yalnız yerel çalışma alanında tutulur ve Git tarafından izlenmez.

## TGSS 2024

TGSS 2024 mikro verisi, İSAR Araştırma Merkezi tarafından yürütülen Türkiye Genel Sosyal Saha Araştırması kapsamında oluşturulmuştur. Veriye erişim ve kullanım, veri sağlayıcının resmî koşullarına tabidir.

Bu depoda aşağıdaki içerikler bulunmaz:

- ham TGSS mikro verisi,
- hazırlanmış satır düzeyi TGSS verisi,
- satır düzeyi tahminler veya artıklar,
- katılımcı kimliği, özgün satır numarası veya kayıt anahtarı,
- yeniden tanımlama riski taşıyan doğrudan veya dolaylı kayıtlar,
- küçük hücre sıklıkları,
- paylaşım izni kesinleşmemiş TGSS ara dosyaları.

Kamuya açık TGSS içeriği şunlarla sınırlıdır:

- kullanılan değişkenlerin analitik rolleri,
- veri hazırlama ve filtreleme kuralları,
- gelir kategorilerinin temsilî değerlere dönüştürülme tablosu,
- yeterli yuvarlama uygulanmış betimsel özetler,
- kat-model düzeyindeki performans sonuçları,
- tezde kullanılan toplulaştırılmış tablo ve şekiller.

TGSS ham verisi yalnız yetkili kullanıcının yerel çalışma alanında `data/raw/tgss/TGSS2024.csv` yolu üzerinden kullanılabilir. Yol, beklenen dizin yapısını gösterir. Dosyanın kendisi depoda bulunmaz.

## Karşılaştırma veri setleri

Karşılaştırma veri setleri özgün sağlayıcılarından edinilmelidir:

- Abalone: UCI Machine Learning Repository
- Concrete Compressive Strength: UCI Machine Learning Repository
- Wine Quality Red: UCI Machine Learning Repository
- Air Quality: UCI Machine Learning Repository
- Servo: UCI Machine Learning Repository
- California Housing: scikit-learn `fetch_california_housing`

Yerel dosya adları ve beklenen dizin yapısı `data/raw/README.md` içinde belgelenmiştir.

## Gönderim öncesi güvenlik kontrolü

GitHub’a gönderimden önce:

1. `data/raw/` altında README dışında dosya bulunmadığı doğrulanır.
2. TGSS adı taşıyan veya kısıtlı veri biçimlerinde saklanan dosyalar taranır.
3. Satır düzeyi çıktı, tahmin, artık ve kayıt anahtarları aranır.
4. Küçük hücre sıklıkları içeren TGSS özetlerinin bulunmadığı doğrulanır.
5. Depo bütünlük ve veri güvenliği testi çalıştırılır.

Bu teknik kontroller, veri sağlayıcının erişim ve kullanım koşullarının yerine geçmez.

