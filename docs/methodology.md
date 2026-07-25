# Yöntem uygulaması

## Kat içi veri işleme

Veri hazırlama ve modelleme adımları `src/imbalanced_regression/` altında iki kanonik modülde uygulanır:

- `data_preparation.py`, veri kaynaklarını analitik veri yapılarına dönüştürür ve grup kimliklerini üretir.
- `fold_pipeline.py`, kat içi ön işleme, ilgililik, yeniden örnekleme, modelleme ve performans ölçümü adımlarını yürütür.

Bu ayrım, ham veri erişimi ile model değerlendirme işlemlerinin birbirinden bağımsız incelenmesini sağlar.

## SMOTE-R ve SMOGN tabanlı modeller

Tezde kullanılan SMOTE-R + RF ve SMOGN + RF etiketleri, ilgili yöntemlerin temel ilkelerini ortak deney protokolüne uyarlayan tez-özel uygulamaları ifade eder. Yeniden örnekleme yalnız eğitim katında gerçekleştirilir. Test verisi hiçbir aşamada sentetik örnek üretimine veya ön işleme parametrelerinin belirlenmesine katılmaz.

## İlgililik fonksiyonu

İlgililik fonksiyonu her eğitim katının hedef dağılımından hesaplanır. Test katının hedef değerleri kontrol noktalarının belirlenmesinde kullanılmaz. Nadir bölge eşiği aynı kat içi ilgililik fonksiyonu üzerinden uygulanır.

## OİHA

Ortak İlgililik Hata Alanı, her gözlem için gerçek ve tahmin edilen hedef değerlerinin ilgililik skorlarından büyük olanını kullanır. Ardışık ilgililik eşiklerinde hesaplanan normalize karesel hata toplamları trapez yöntemiyle bütünleştirilir.

Bu gösterge:

- gerçek hedefi yüksek ilgililik taşıyan gözlemlerdeki hataları,
- tahmini yüksek ilgililikli bir bölgeye taşınan gözlemlerdeki hataları

birlikte görünür kılar. Düşük OİHA değeri daha iyi performansı gösterir.

## Sonuçların yorumu

Model seçimi tek bir performans ölçütüne dayandırılmaz. Genel hata ölçütleri ile ilgililik temelli ölçütler birlikte değerlendirilir. Bu yaklaşım, merkezi hedef bölgesindeki başarı ile seyrek hedef bölgelerindeki başarı arasındaki değiş tokuşu görünür kılar.
