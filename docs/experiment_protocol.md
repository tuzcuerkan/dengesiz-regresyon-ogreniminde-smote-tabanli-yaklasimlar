# Deney protokolü

## Veri setleri

Çalışmada altı karşılaştırma veri seti ile TGSS 2024 verisinden türetilen iki uygulama veri seti kullanılmıştır:

- Abalone
- California Housing
- Concrete Compressive Strength
- Wine Quality Red
- Air Quality NO₂
- Servo
- TGSS beden kitle indeksi
- TGSS yaklaşık kişisel aylık net gelir

## Modeller

- En Küçük Kareler
- Rastgele Orman
- SMOTE-R + Rastgele Orman
- SMOGN + Rastgele Orman
- İlgililik Ağırlıklı Rastgele Orman

## Ana değerlendirme düzeni

Ana deney, grup kontrollü iki tekrarlı 5 katlı çapraz doğrulama ile yürütülmüştür. Aynı açıklayıcı değişken vektörüne sahip kayıtlar aynı grup kimliği altında tutulmuş ve hiçbir grup eğitim ile test kümelerinde birlikte yer almamıştır.

Her eğitim katında:

1. Eksik değer tamamlama parametreleri öğrenilir.
2. Sayısal değişkenlerin ölçekleme parametreleri öğrenilir.
3. İlgililik fonksiyonu yalnız eğitim hedeflerinden hesaplanır.
4. Yeniden örnekleme yalnız eğitim verisine uygulanır.
5. Model eğitilir ve ayrılmış test katında değerlendirilir.

## Performans ölçütleri

Genel tahmin başarısı RMSE, MAE ve R² ile değerlendirilmiştir. Yüksek ilgililik taşıyan hedef bölgelerindeki davranış WMSE, SERA, OİHA, nadir bölge RMSE ve normal bölge RMSE ile incelenmiştir.

OİHA, gerçek ve tahmin edilen hedef değerlerinin ortak ilgililiğini kullanan tez-özel tamamlayıcı göstergedir.

## Duyarlılık analizleri

- TGSS BMI değişken şeması duyarlılığı
- Air Quality açıklayıcı değişken kapsamı duyarlılığı
- Yinelenen açıklayıcı vektörlerde standart KFold karşılaştırması
- Grup kontrollü tek 10 katlı çapraz doğrulama

Duyarlılık analizleri ana model seçimi için kullanılmamış, ana bulguların veri hazırlama ve kat sayısı kararlarına bağlılığını incelemek amacıyla raporlanmıştır.

## Değerlendirme kapsamı

| Analiz | Kat-model değerlendirmesi |
|---|---:|
| Ana deney | 400 |
| Veri hazırlama duyarlılıkları | 250 |
| Tek 10 kat duyarlılığı | 400 |
| Bağımsız yeniden üretim | 400 |
| Toplam | 1.450 |
