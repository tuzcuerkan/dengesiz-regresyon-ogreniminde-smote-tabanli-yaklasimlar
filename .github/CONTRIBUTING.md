# Katkı Rehberi

Bu depo, “Dengesiz Regresyon Öğreniminde SMOTE Tabanlı Yaklaşımların
Karşılaştırılması” başlıklı yüksek lisans tezinin akademik yeniden üretilebilirlik
paketidir. Hata bildirimleri, yeniden üretilebilirlik soruları ve bilimsel
iyileştirme önerileri memnuniyetle karşılanır.

## Bildirimden önce

- `README.md` dosyasını inceleyin.
- Yeniden üretim adımları için `docs/reproducibility.md` belgesine bakın.
- Veri erişimi ve paylaşım sınırları için
  `docs/data_access_and_privacy.md` belgesini okuyun.
- Mevcut Issues kayıtlarında aynı konunun daha önce bildirilip bildirilmediğini
  kontrol edin.

## Bildirim türleri

- Bir komut, beklenen çıktı veya yeniden üretim adımıyla ilgili sorular için
  **Yeniden üretilebilirlik sorusu** formunu kullanın.
- Kodun veya doğrulama aracının beklenmeyen davranışı için **Hata bildirimi**
  formunu kullanın.
- Güvenlik, erişim anahtarı veya veri gizliliği sorunlarını herkese açık bir
  Issue olarak paylaşmayın. Deponun özel güvenlik bildirimi özelliğini kullanın.

## Veri gizliliği

Issue, pull request, commit veya ek dosya içerisinde aşağıdaki içerikleri
paylaşmayın:

- TGSS 2024 ham veya hazırlanmış satır düzeyi mikro verileri,
- katılımcı kimlikleri veya özgün kayıt numaraları,
- kişisel veya yeniden tanımlamaya elverişli bilgiler,
- erişim anahtarları, parolalar, tokenlar veya `.env` içerikleri,
- yeniden dağıtımı kısıtlanan üçüncü taraf veri dosyaları.

Sorunu açıklamak için gerekli olduğunda anonimleştirilmiş, yapay veya yeterince
toplulaştırılmış küçük bir örnek kullanın.

## Kod katkıları

1. Depoyu fork edin ve değişikliğiniz için ayrı bir dal oluşturun.
2. Değişikliği mümkün olduğunca dar kapsamlı tutun.
3. Python 3.13.5 ortamında gerekli bağımlılıkları kurun.
4. Aşağıdaki doğrulamayı çalıştırın:

   ```bash
   python tests/validate_repository.py
   ```

5. Değişikliğin amacı, etkilediği dosyalar ve doğrulama sonucu açıklanmış bir
   pull request oluşturun.

Bilimsel sonuçları etkileyen önerilerde kullanılan veri kapsamı, deney protokolü,
tohumlar, değerlendirme ölçütleri ve beklenen etki açıkça belirtilmelidir.

## Lisans

Kaynak kod katkıları MIT Lisansı, belge ve araştırma çıktısı katkıları CC BY 4.0
kapsamında değerlendirilir. Üçüncü taraf veri setleri ile TGSS mikro verisi bu
lisansların kapsamına girmez.
