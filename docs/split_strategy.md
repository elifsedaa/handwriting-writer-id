# Bölme stratejisi (Gün 14)

- **Önce sayfayı böl, sonra patch/satır çıkar.** Aynı sayfadan çıkan hiçbir parça iki bölmede bulunmaz.
- `split_closed` (closed-set sınıflandırma): her yazarın sayfaları train/val/test'e dağıtılır (~%70/15/15, val ve test en az 1 sayfa).
  Sayfası 3'ten az olan yazarlar `excluded` olur (örn. tek formlu IAM yazarları).
- `split_open` (open-set / metric learning): yazar bazında bölünür; test yazarları eğitimde hiç görülmez. Veri setleri ayrı ayrı bölünür.
- Seed 42, `numpy.random.default_rng`. Sonuç `data/metadata/all_samples.csv` içindedir ve repoda tutulur (kimlik bilgisi yok).
- Her çalıştırmada sızıntı denetimi yapılır (`check_leakage`).
- Notlar: [ ] (gerçek yazar/sayfa sayıları geldikten sonra doldur)
