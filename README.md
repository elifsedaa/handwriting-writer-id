# El Yazısından Yazar Tanıma

El yazısı görüntülerinden yazının hangi kişiye ait olduğunu belirleyen derin öğrenme projesi.
Amaç yazının içeriğini okumak değil, kişiye özgü yazı özelliklerinden yazarı tanımaktır.

## Kurulum
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
# Önce pytorch.org'daki komutla torch + torchvision'ı kur, sonra:
pip install -r requirements.txt
```

## Klasörler
- `data/raw`, `data/processed`, `data/metadata`: veri (git'e eklenmez, metadata CSV hariç tutulabilir)
- `src/`: ortak kod (`utils`, `config`, `experiment`, `viz`, `data/`, `models/`)
- `scripts/`: günlük çalışma betikleri
- `experiments/figures/`: grafikler, `experiments/runs/`: deney kayıtları
- `docs/`: literatür, veri seti, protokol, form ve plan notları

## Günlük betikler (proje kök klasöründen çalıştır)
| Gün | Komut | Ne yapar |
|---|---|---|
| 2 | `python scripts/day02_nn_anatomy.py` | Sinir ağı anatomisi |
| 3 | `python scripts/day03_check_env.py` | Ortam/GPU kontrolü |
| 4 | `python scripts/day04_tensors.py` | Tensör oluşturma, veri tipleri |
| 5 | `python scripts/day05_tensor_ops.py` | Öznitelikler, matmul, nn.Linear |
| 5 | `python scripts/day05_gantt.py` | Proje Gantt şeması |
| 6 | `python scripts/day06_shapes_aggregation.py` | Shape hataları, min/max/mean/argmax |
| 6 | `python scripts/day06_check_datasets.py` | Veri seti durum raporu (`docs/dataset_status.md`) |
| 7 | `python scripts/day07_reshape_indexing.py` | reshape/permute/indexing, patch çıkarma |
| 8 | `python scripts/day08_numpy_repro_device.py` | NumPy, seed, cihaz, deney klasörü |
| 9 | `python scripts/day09_linear_data.py` | Doğrusal regresyon verisi + grafik |
| 9 | `python scripts/day09_fundamentals_exercises.py` | Fundamentals alıştırmaları |
| 10 | `python scripts/day10_first_model.py` | İlk nn.Module, state_dict, inference_mode |
| 11 | `python scripts/day11_setup_iam.py` | IAM arşivlerini aç + doğrula |
| 11 | `python scripts/day11_training_loop.py` | Eğitim döngüsü (doğrusal regresyon), loss eğrisi |
| 12 | `python scripts/day12_eda.py` | EDA grafikleri (yazar sayısı, örnek dağılımı, görüntü boyutu) |
| 13 | `python scripts/day13_build_metadata.py` | `data/metadata/all_samples.csv` üretir |
| 13 | `python scripts/day13_save_load.py` | Model kaydet/yükle (state_dict) |
| 14 | `python scripts/day14_make_splits.py` | `split_closed` (sayfa) + `split_open` (yazar) sütunları |
| 14 | `python scripts/day14_toy_classification.py` | İki daire verisi, girdi/çıktı şekilleri, grafik |
| 15 | `python scripts/day15_classification_nn.py` | Tensöre çevirme, bölme, küçük ağ (nn.Module / nn.Sequential) |
| 15 | `python scripts/day15_preprocess_demo.py [yol]` | Ön işleme hattı önce/sonra grafiği |
| 16 | `python scripts/day16_classification_loop.py` | Kayıp, logits→olasılık→etiket, sınıflandırma döngüsü |

| 16 | `python scripts/day16_crop_patches_demo.py [yol]` | Kırpma + patch çıkarma, önce/sonra grafikleri |
| 17 | `python scripts/day17_make_patches.py` | Patch veri seti (`data/processed/patches/`, `patches_index.csv`) |
| 17 | `python scripts/day17_improve_model.py` | Daha çok katman/gizli birim, karar sınırı grafikleri |
| 17 | `python scripts/day17_straight_line.py` | Düz çizgi verisi ile hata ayıklama |
| 18 | `python scripts/day18_nonlinearity.py` | ReLU'lu model, saf PyTorch aktivasyonlar |
| 18 | `python scripts/day18_report.py` | `docs/ilk_18_gun_ozeti.md` (sayılar gerçek dosyalardan) |

Testler (torch gerektirmeyen kısımlar, sentetik veriyle): `python tests/test_week3.py` ve `python tests/test_week4.py`
