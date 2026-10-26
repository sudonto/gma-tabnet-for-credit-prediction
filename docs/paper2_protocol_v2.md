# Paper 2 — Protokol evaluasi putaran kedua (dideklarasikan sebelum run)

Ditulis 2026-10-04, sebelum satu pun run putaran kedua dijalankan pada pc-server.
Perubahan setelah tanggal ini harus dicatat di bagian akhir beserta alasannya.

Dokumen ini menjawab komentar reviewer IJIES (Paper ID 20266222). Semua keputusan
di bawah dibuat sebelum hasil dilihat, sehingga apa pun hasilnya dilaporkan apa adanya.

## Evaluasi luar

- 3 ulangan × 5 lipatan stratified CV = 15 lipatan luar per dataset
  (`src/common/splits.py: repeated_stratified_splits`). Ulangan r memakai
  `StratifiedKFold(shuffle=True, random_state=r)`.
- Di dalam partisi latih tiap lipatan: blok validasi bertingkat 15%
  (early stopping, ambang F1, dan skor pencarian hiperparameter).
- Seed model per lipatan = `1000·r + k`. Seed pengoptimasi berasal dari
  `SeedSequence([r, k, id_optimiser, id_arsitektur])`.
- Lipatan uji hanya disentuh sekali, untuk menilai model akhir. Tidak ada
  konfigurasi yang dipakai bersama antar lipatan.

## Seleksi hiperparameter

- Aturan seleksi hilir: **TPE**. Ditetapkan di muka; tidak memilih yang terbaik dari empat
  pengoptimasi, karena itu menghabiskan 4× anggaran dan menggelembungkan skor validasi.
- GMA-TabNet dan TabNet vanilla masing-masing dicari dengan TPE pada anggaran yang sama.
- Anggaran per pencarian: German 600 s, UCI Taiwan 1200 s, GMSC 1200 s.
  - Batas waktu keras: evaluasi yang terpotong dicatat tetapi tidak dihitung.
  - GPU dipanaskan sebelum jam mulai berjalan.
- Pencarian dilatih pada ≤ 15.000 baris latih (jadwal 25 epoch, patience 5).
  Konfigurasi terpilih dilatih ulang pada seluruh baris latih (120 / 20).
- Studi pengoptimasi (random, TPE, PSO, GA):
  - dijalankan pada ulangan 0 (5 lipatan), dengan pencarian independen per lipatan;
  - konfigurasi terpilih dinilai pada lipatan uji.

## Lengan eksperimen

Lengan *fixed-config* memakai konfigurasi GMA lipatan tersebut:

- `gma_full`
- `tabnet_vanilla`
- `gma_single_head`
- `gma_fixed_temperature`
- `gma_no_interaction_gate`
- `gma_no_focal`
- `gma_uniform_mix`
- `gma_pretrained`
- `tabnet_vanilla_matched` (jumlah parameter ≈ GMA)
- `mh_wang_repro`

*Equal-tuning*: `tabnet_vanilla_tuned`.

Rantai dekomposisi selisih TabNet:

1. `tabnet_reference` (setelan pengajuan pertama)
2. `ref_lib_tuned`
3. `ours_refcompat_tuned`
4. `ours_vanilla_tuned_bce`
5. `tabnet_vanilla_tuned`

Pembanding:

- 10 model default;
- XGBoost, LightGBM, RF dan SVM yang disetel dengan TPE, dengan jumlah trial tetap.

Stacking:

- `stack_gma_nested`
- `stack_vanilla_nested`
- `stack_no_extractor`
- `stack_gma_shared` (protokol lama)

Inner fold: K = 5 (German, UCI) dan K = 3 (GMSC).

## Inferensi

- Unit inferensi: lipatan luar (J = 15).
- Uji: selisih berpasangan dengan uji-t tersampel ulang terkoreksi (Nadeau–Bengio),
  dengan rasio n_uji / n_latih_luar ≈ 0,25 dan SK 95%.
- Koreksi Holm diterapkan per famili:

| Famili | Perbandingan | Lingkup Holm |
|---|---|---|
| F1 (utama) | GMA vs vanilla-tuned | lintas 3 dataset |
| F2 | lengan mekanisme/kontrol vs GMA | per dataset |
| F3 | stack GMA vs {stack vanilla, tanpa ekstraktor, pembanding terkuat} | — |
| F4 | TPE/PSO/GA vs acak, AUC uji, J = 5 | per dataset |
| F5 | nested vs shared | deskriptif |

- Uji yang tidak dipakai lagi:
  - Friedman/Nemenyi: tidak dipakai sebagai bukti, karena 3 dataset tidak punya daya uji.
  - DeLong/bootstrap pada prediksi gabungan lintas lipatan: tidak dipakai.
- Ketiadaan perbedaan signifikan **tidak** ditulis sebagai kesetaraan. Kesetaraan hanya
  boleh diklaim dengan TOST terkoreksi bermargin ±0,005 AUC.
- Pembanding terkuat = model dengan AUC rata-rata tertinggi di antara 14 pembanding
  (konservatif terhadap model usulan).

## Atribusi

- Lingkup: ulangan 0, 5 lipatan, ≤ 4.000 baris uji per lipatan.
- Metode:
  - GMA: mask berbobot keputusan, IG, permutasi;
  - XGBoost: TreeSHAP, permutasi;
  - stack: permutasi.
- Stabilitas:
  - Kendall W antar lipatan;
  - ρ dan Jaccard@3 terhadap gangguan input (noise 0,05 sd; 5% kode kategori ditukar).
- Alasan per pemohon: 3 teratas pada desil risiko tertinggi.

## Sensitivitas protokol (GMSC)

- Protokol Wang & Zhang (2025): NearMiss-2 sebelum split 4:1, dengan 2 fitur dibuang.
- Pembanding: NearMiss-2 hanya pada data latih, dan tanpa penyeimbangan.
- Model: XGBoost dan GMA-TabNet, 5 seed.

## Log perubahan

(kosong)
