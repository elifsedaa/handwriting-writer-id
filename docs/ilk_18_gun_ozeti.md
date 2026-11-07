# İlk 18 gün özeti (otomatik üretildi 2026-10-22 01:51)

Bilinmeyen değerler `[ ]` olarak bırakıldı; elle doldurulacak.

## 1. Veri durumu

| Veri seti | Ham görüntü | Yazar | Sayfa | Örnek (metadata) |
|---|---|---|---|---|
| iam | 115319 | [ ] | [ ] | [ ] |
| cvl | 1604 | 310 | 1604 | 1604 |
| firemaker | [ ] | [ ] | [ ] | [ ] |

IAM başvuru/onay durumu: [ ]

## 2. Bölme

closed-set (sayfa bazında) örnek sayıları:

```
split_closed  test  train  val
dataset                       
cvl            310    984  310
```

open-set (yazar bazında) yazar sayıları:

```
split_open  test  train  val
dataset                     
cvl           46    218   46
```

## 3. Ön işleme raporu

- Patch boyutu: 128x128 (config.py), ölçek 0.5 (≈150 dpi), boş patch eşiği: koyu piksel oranı < 0.02
- Sıra: gri tonlama -> kenar kırpma -> yazı bölgesine kırpma -> ölçekleme -> patch -> boş patch eleme
- Satır görüntüsü hattı (preprocess_line): 64x512, beyaz doldurma, [0,1]
- Yatay çevirme gibi el yazısını bozan artırma kullanılmıyor
- Kural: önce sayfa bölünür, sonra patch çıkarılır

Üretilen patch: **44654** | yazar: 310 | sayfa: 1602 | ortalama koyu piksel oranı: 0.047

```
split    test  train   val
dataset                   
cvl      8635  27329  8690
```

## 4. Çalıştırılan kod

- Torch gerektirmeyen kısımlar: `python tests/test_week3.py`, `python tests/test_week4.py`
- Torch'lu betiklerin çalıştırma durumu: [ ]

## 5. Açık sorunlar / sonraki adımlar

- [ ]
