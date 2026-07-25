# Yeniden üretim kılavuzu

## Gereksinimler

- Python 3.13.5
- `requirements.txt` içinde belirtilen paketler
- Özgün sağlayıcılarından edinilmiş yerel veri dosyaları
- TGSS uygulamaları için yetkili TGSS 2024 mikro veri erişimi

## Ortam kurulumu

```bash
python -m venv .venv
```

Linux ve macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Veri yerleşimi

Beklenen yerel dizin yapısı `data/raw/README.md` içinde verilmiştir. Alternatif kök dizin `THESIS_RAW_DATA_DIR` ortam değişkeniyle tanımlanabilir.

## Çalıştırma sırası

Ana deney ve veri hazırlama duyarlılıkları:

```bash
python scripts/run_main_and_data_sensitivity.py
```

Grup kontrollü tek 10 kat duyarlılığı:

```bash
python scripts/run_protocol_sensitivity_10fold.py
```

Bağımsız yeniden üretim:

```bash
python scripts/run_independent_reproduction.py
```

Çalışma dosyaları `.runs/` altında oluşturulur ve Git tarafından izlenmez.

## Beklenen doğrulamalar

- Ana deney: 400 değerlendirme
- Veri hazırlama duyarlılıkları: 250 değerlendirme
- Tek 10 kat duyarlılığı: 400 değerlendirme
- Bağımsız yeniden üretim: 400 değerlendirme
- Model birinciliği uyumu: 88/88
- Tam sıralama uyumu: 88/88
- Mutlak tolerans: `1e-10`
- Göreli tolerans: `1e-12`

## Depo bütünlüğü

```bash
python tests/validate_repository.py
```

Bu komut dosya bütünlüğünü, sonuç sayılarını, OİHA terminolojisini, yeniden üretim kayıtlarını ve kısıtlı veri bulunmadığını denetler.

