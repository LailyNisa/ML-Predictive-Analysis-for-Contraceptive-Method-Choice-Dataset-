# Laporan Proyek Machine Learning Terapan - Submission 1 _Predictive Analytics_ -

### _Contraceptive Method Classification Dataset_

Laily Khoirunnisa' - MLT4

## Domain Proyek

Pemerintah telah melakukan berbagai program untuk menekan laju pertambahan
penduduk. Salah satu upaya pengendalian laju pertumbuhan penduduk yang paling efektif adalah dengan penggunaan alat kontrasepsi untuk menghindari “4 terlalu” seperti terlalu tua, terlalu muda, terlalu banyak anak, dan terlalu dekat jarak kelahiran (Budijanto, 2013). Pengendalian laju pertambahan jumlah penduduk perlu dilakukan agar tidak terjadi ledakan penduduk (Asih dan Oesman, 2009).
Pada tahun 1987, dilakukan survey kontrasepsi nasional yang sampelnya adalah wanita yang sudah menikah baik yang belum hamil atau tidak tahu apakah mereka hamil pada saat wawancara. Tujuan survey adalah untuk memprediksi pilihan metode kontrasepsi seorang wanita (tidak pakai kontrasepsi, metode jangka panjang, atau metode jangka pendek) berdasarkan pada karakteristik demografis dan sosial ekonominya.

## Business Understanding

Penelitian ini bertujuan untuk menganalisis faktor apa saja yang mempengaruhi pemilihan metode kontrasepsi seorang wanita. Dengan demikian, pemerintah bekerja sama dengan perusahaan kontrasepsi dapat menentukan objek sasaran bantuan sekaligus sasaran pasar dengan lebih tepat.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:

- Apa saja faktor dari sisi seorang wanita yang paling mempengaruhi pilihan metode kontrasepsi?
- Bagaimana prediksi pilihan metode kontrasepsi pada seorang wanita

### Goals

Menjelaskan tujuan dari pernyataan masalah:

- Menentukan faktor yang paling berpengaruh pada diri seorang wanita dalam memilih metode kontrasepsi.
- Memprediksi pilihan metode kontrasepsi yang umum digunakan sesuai faktor-fakor yang mempengaruhi keputusan pilihan metode seorang wanita,

  ### Solution statements

  - Setelah mengetahui fitur mana yang sangat berpengaruh terhadap faktor pilihan kontrasepsi, maka akan dilanjutkan dengan komparasi antar algoritma klasifikasi Decision Tree, Random Forest dan AdaBoost, dengan tambahan hyperparameter tuning untuk mendapat hasil yang optimal.
  - Hasil yang diberikan kemudian dilakukan evaluasi dan korelasi metrik dengan metrik precistion, recall, F1-score dan ROC-AUC.

## Data Understanding

1. Judul: Contraceptive Method Choice
2. Sumber data:
   (a) Asal data: Dataset ini adalah sebuah subset dari Survey National Prevalensi Kontrasepsi pada tahun 1987 di Indonesia
   (b) Kreator: Tjen-Sien Lim (limt@stat.wisc.edu)
   (c) Donor: Tjen-Sien Lim (limt@stat.wisc.edu)
   (d) Tanggal: June 7, 1997

Link Kaggle dataset yang digunakan adalah sebagai berikut: [Contraceptive Method Choice](https://www.kaggle.com/datasets/faizunnabi/contraceptive-method-choice). Data juga terdapat di website [UCI Machine Learning](Source: https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice)

### Variabel-variabel pada 'Contraceptive Method Choice' Kaggle dataset adalah sebagai berikut:

- Wife's age/Usia Istri (numerik)
- Wife's education/Pendidikan Istri (kategorikal) 1=SD, 2=SMP, 3=SMA, 4=Perguruan Tinggi
- Husband's education/Pendidikan suami (kategorikal) 1=SD, 2=SMP, 3=SMA, 4=Perguruan Tinggi
- Number of children ever born/Jumlah anak (numerik)
- Wife's religion/Agama Istri (binary) 0=Non-Islam, 1=Islam
- Wife's now working?/Istri bekerja (binary) 0=Ya, 1=Tidak
- Husband's occupation/Pekerjaan suami (kategorikal) 1=tidak bagus, 2=cukup bagus, 3=bagus, 4=sangat bagus
- Standard-of-living index/Indeks standar hidup (kategorikal) 1=sangat rendah, 2=rendah, 3=menengah, 4=tinggi
- Media exposure/tekanan media (binary) 0=Bagus, 1=Tidak Bagus
- Contraceptive method used/Metode Kontrasepsi (kelas atribut) 1=Tidak pakai, 2=Jangka panjang, 3=Jangka pendek

Pada tahap ini, dilakukan teknik visualisasi data atau exploratory data analysis (EDA).

##### 1. Memetakan jenis atribut, tipe data, dan distribusinya

```sh
    RangeIndex: 1473 entries, 0 to 1472
    Data columns (total 10 columns):
         Column                Non-Null Count  Dtype
    ---  ------                --------------  -----
     0   Usia Istri            1473 non-null   int64
     1   Pend. Istri           1473 non-null   int64
     2   Pend. Suami           1473 non-null   int64
     3   Jumlah Anak           1473 non-null   int64
     4   Agama Istri           1473 non-null   int64
     5   Istri Bekerja         1473 non-null   int64
     6   Pekerjaan Suami       1473 non-null   int64
     7   Indeks Standar Hidup  1473 non-null   int64
     8   Tekanan Media         1473 non-null   int64
     9   Metode Kontrasepsi    1473 non-null   int64
    dtypes: int64(10)
```

Dari output data terlihat bahwa:

- Jumlah data: 1473
- Jumlah atribut: 10 (termasuk kelas atribut). Dengan 9 variabel fitur > > dan 1 variabel kelas yang merupakan target fitur, yaitu 'Metode Kontrasepsi'.
- Semua data ditampilkan dengan tipe data numerik integer, data kategorikal dan binary telah direpresentasikan dengan level angka.

##### Analisis Deskriptif

```sh
        UI      PI     PS     JA     AI    IB     PS    ISH    TM     MK
count  1473    1473   1473   1473   1473  1473   1473   1473   1473  1473
mean  32.54    2.9    3.43   3.26   0.85  0.75	 2.14   3.13   0.07  1.92
std    8.23    1.01   0.82   2.36   0.36  0.43   0.86   0.98   0.26	 0.88
min	  16.0	   1.0	  1.0	 0.00   0.00  0.00	 1.00   1.00   0.0   1.00
25%	  26.0     2.0    3.0    1.0    1.0   0.0	 1.0    3.0	   0.0	 1.0
50%   32.0     3.0	  4.0	 3.0	1.0	  1.0	 2.0    3.0	   0.0	 2.0
75%   39.0	   4.0    4.0	 4.0    1.0	  1.0	 3.0	4.0	   0.0	 3.0
max   49.0	   4.0	  4.0 	16.0	1.0	  1.0	4.0     4.0	   1.0	 3.0
```

> UI = Usia Istri, PI = Pend. Istri, PS = Pend. Suami, JA = Jumlah Anak, AI = Agama Istri, IB = Istri Bekerja, PS = Pekerjaan Suami, ISH = Indeks Standar Hidup, TM = Tekanan Media, MK = Metode Kontrasepsi

Dari data di atas, dapat disimpulkan :
• `'Usia Istri'` pada rentang 16-49 tahun, dengan rata-rata berusia 32-33 tahun
• `'Pend. Istri'` pada rentang 1-4, dengan rata-rata 2.9 atau pendidikan setingkat SMA
• `'Pend. Suami'` pada rentang 1-4 dengan rata-rata 3.43 atau berpendidikan antara SMA-Perguruan Tinggi
• `'Jumlah Anak'` pada rentang 0-16, dengan rata-rata 3.26 atau memiliki 3-4 anak
• `'Agama Istri'` pada rentang 0-1 dengan rata-rata 0.85 atau beragama Islam
• `'Istri Bekerja'` pada rentang 0-1 dengan rata-rata 0.75, atau status istri mempunyai pekerjaan
• `'Pekerjaan Suami'` pada rentang 1-4 dengan rata-rata 2.14, atau memiliki standar pekerjaan yang cukup bagus
• `'Indeks Standar Hidup'` pada rentang 1-4, dengan rata-rata 3.13, atau memiliki standar hidup level menengah
• `'Tekanan Media'` pada rentang 0-1, dengan rata-rata 0.07, atau media hampir tidak berperan dalam keputusan kontrasepsi seorang istri
• `'Media Kontrasepsi'` pada rentang 1-4, dengan rata-rata 1.92, atau menggunakan kontrasepsi

##### Data Numerikal

- **Grafik Jumlah `Usia Istri` per Target Kelas `Metode Kontrasepsi`**
  ![image](https://drive.google.com/uc?export=view&id=1uHOlN4v_RBqRBhS89VOgBBMkVmdv5ReC)

- **Grafik Jumlah `Jumlah Anak` per Target Kelas `Metode Kontrasepsi`**
  ![image](https://drive.google.com/uc?export=view&id=1k3AkZqlGAFVlvHah40PR1KeEZ6WXRICp)

- Dari 2 grafik data numerikal, dapat dilihat bahwa setiap kelas hampir memiliki sebaran data yang sama, yang membedakan hanya jumlah datanya saja.
- Pada grafik `Usia Istri`, rentang usia 16-49 tahun, urutan paling banyak pada kategori kelas: `tidak pakai kontrasepsi-pakai jangka pendek-pakai jangka panjang`
- Pada grafik `Jumlah Anak`, rentang usia 0-16 anak. Urutan jumlah `banyak anak` per kategori kelas dari yang tertinggi: `tidak pakai kontrasepsi-pakai jangka pendek-pakai jangka panjang`

##### Data Kategorikal

```sh
    Jumlah data kelas 'Pend. Istri' :
    Kategori    jumlah sampel  persentase
    4            577            39.2
    3            410            27.8
    2            334            22.7
    1            152            10.3
```

> Ket : 1=SD, 2=SMP, 3=SMA, 4=Perguruan Tinggi

```sh
    Jumlah data kelas 'Pend. Suami' :
    Kategori  jumlah sampel  persentase
    4            899            61.0
    3            352            23.9
    2            178            12.1
    1             44             3.0
```

> Ket : 1=SD, 2=SMP, 3=SMA, 4=Perguruan Tinggi

```sh
    Jumlah data kelas 'Agama Istri':
    Kategori  jumlah sampel  persentase
    1           1253          85.1
    0            220          14.9
```

> Ket : 0=Non-Islam, 1=Islam

```sh
    Jumlah data kelas 'Istri Bekerja' :
    Kategori  jumlah sampel  persentase
    1           1104         74.9
    0            369         25.1
```

> Ket : 0=Tidak, 1=Bekerja

```sh
    Jumlah data kelas 'Pekerjaan Suami' :
    Kategori  jumlah sampel  persentase
        3            585        39.7
        1            436        29.6
        2            425        28.9
        4             27         1.8
```

> Ket : 1=tidak bagus, 2=cukup bagus, 3=bagus, 4=sangat bagus

```sh
    Jumlah data kelas 'Indeks Standar Hidup' :
    Kategori  jumlah sampel  persentase
        4            684        46.4
        3            431        29.3
        2            229        15.5
        1            129         8.8
```

> Ket : 1=sangat rendah, 2=rendah, 3=menengah, 4=tinggi

```sh
    Jumlah data kelas 'Tekanan Media' :
    Kategori  jumlah sampel  persentase
        0           1364        92.6
        1            109         7.4
```

> Ket : 0=Tidak, 1=Ya

```sh
    Jumlah data kelas Metode Kontrasepsi :
    Nama Kelas     jumlah sampel  persentase
        1            629            42.7
        3            511            34.7
        2            333            22.6
```

> Ket : 1=Tidak pakai, 2=jangka panjang, 3=jangka pendek

Dapat kita lihat di sini, bahwa jumlah sampel antar target kelas terdapat perbedaan, data seperti ini disebut imbalance data. Agar prediksi lebih baik, maka perlu ditambahkan sampling pada tahap data preparation. Metode yang akan digunakan yaitu menggabungkan 2 metode oversampling dan undersampling dengan menggunakan algoritma SMOTE dan Tomek.

![image](https://drive.google.com/uc?export=view&id=1qTJ7KMYNvDBJsdcaov_lfoLO2jAlo144)

#### 2. Deteksi missing value (Nulls Check)

```sh
    print("Value kosong pada data :",df.isnull().values.any())
    Value kosong pada data : False
```

> Artinya, tidak ada value kosong pada data.

#### 3. Memetakan korelasi antara fitur dan target

```sh
    Params                  Value
    Usia Istri              -0.16
    Pend. Istri              0.15
    Pend. Suami              0.1
    Jumlah Anak              0.08
    Agama Istri             -0.03
    Istri Bekerja            0.05
    Pekerjaan Suami          0.02
    Indeks Standar Hidup     0.09
    Tekanan Media           -0.12
```

Korelasi fitur pada data CMC, paling tinggi hanya sebesar 0.15 dari `Pend. Istri`. Sedangkan korelasi negatif paling dominan yakni `Usia Istri`. Korelasi yang paling kecil, yaitu 0.02 dari parameter `Pekerjaan Suami` dan nilai -0.03 dari `Agama Istri`. Seleksi data yang dilakukan, yaitu dengan menghapus kedua fitur ini.

## Data Preparation

Setelah menganalisa data, dataset CMC tergolong imbalance data. Algoritma dan metode yang akan dilakukan untuk data preparation, antara lain Train-Test-Split untuk pembagian data, teknik StandarScaler untuk standarisasi, SMOTE untuk oversampling, Tomek Links untuk undersampling, dan Pipeline untuk menggabungkan hasil SMOTE dan Tomek. Penjelasan tahap data preparation sebagai berikut:

1. Encoding fitur kategori.
   Sebelum dilakukam encoding, data kategori pada dataset CMC telah menjadi numerik dengan tipe integer. Jadi, dapat langsung melannjutkan proses ke tahap berikutnya.
2. Penanganan Outliers
   Outliers telah dilihat melalui grafik di tahap analisa data. Data outlier terdapat pada data numerik `Jumlah Anak`, dan tidak dilakukan penghapusan outlier karena jumlah yang sedikit dan korelasinya fitur kurang dari 0.1 dengan target kelas.
3. Penghapusan data duplikat
4. Pembagian data tes dan data uji dengan Train-Test-Split
5. Standarisasi dengan StandarScaler.

```sh
    Hasil Standarisasi Fitur Numerik Dataset CMC
    	    Usia Istri	Jumlah Anak
    count	1140.0000	1140.0000
    mean	0.0000	    0.0000
    std	    1.0004	    1.0004
    min	    -2.0106	    -1.3825
    25% 	-0.7907	    -0.5464
    50%	    -0.0587	    -0.1284
    75%	    0.7952  	0.7078
    max	    2.0151	    5.3065
```

5. Melakukan teknik sampling menggunakan kombinasi Over-sampling dan under-sampling (SMOTE dan Tomek)
   Ada beberepa metode oversampling dan under sampling yang telah terbukti secara bersamaan membentuk teknik resampling yang efektif. Salah satunya adalah metode SMOTE dan Tomek links. (Brownlee, Jason.2020)
   **Over-sampling: SMOTE**
   SMOTE merupakan teknik oversampling yang menyeimbangkan data dengan membuat instance sintetik untuk kelas minoritas. Tenik ini bekerja dengan caram mengeliminasi kelas yang sudah ada dengan secara acak dengan mengambil satu titik dari suatu kelas dan menghitung nilai k-neighbor terdekat. Titik sintek ditambahkan antara titik yang dipilih dengan tetangganya. (Dina Elreedy & Amir F.,2019)
   **Under-sampling: Tomek links**
   Tomek links memasangkan dari instance yang sangat dekat, tetapi dari kelas yang berlawanan. Menghapus instance dari kelas mayoritas dari
   setiap pasangan serta meningkatkan ruang antara dua kelas, sehingga dapat membantu proses klasifikasi (Hartayuni Sain & Santi Wulan Purnami,2015).
   Tomek Links merupakan metode undersampling yang menghapus data dari kelas mayoritas yang memiliki karakteristik yang serupa. Namun Tomek Links hanya menghapus instance yang didefinisikan sebagai "Tomek Links" sehingga data yang dianalisis tidak dapat seimbang dan dalam penerapannya metode tersebut dikombinasikan dengan metode SMOTE [14].Teknik ini akan digunakan untuk meresample terhadap kelas minoritas.
   **Oversampling/undersampling sebelum atau setelah splitting data?**
   Aturan praktisnya adalah: jangan pernah mengacaukan set data pengujian. Selalu pisahkan menjadi data uji dan data latih sebelum mencoba teknik oversampling/undersampling.
   Oversampling yang dilakukan sebelum pembagian data dapat memungkinkan pengamatan yang sama persis baik pada data tes maupun data latih. Hal ini memungkinkan model untuk hanya mengingat poin data tertentu dan menyebabkan overfitting dan generalisasi yang buruk pada data uji. Kebocoran data dapat menyebabkan Anda membuat model prediksi yang terlalu optimis dan juga tidak valid. (Rutecki,Marcin.2023)

6. Implementasi Pipeline
   Data hasil oversampling dan undersampling perlu disatukan. 2 transformasi data tersebut dapat disatukan bersama menjadi Pipeline (Brownie, Jason.2021). Pipeline akan membantu menentukan apa, bagaimana, dan di mana data-data akan dikumpulkan. Proses ekstraksi, transformasi, validasi, dan kombinasi data dilakukan secara otomatis menggunakan pipe line. Nantinya data akan dilakukan visualisasi dan analisis lebih lanjut [4].

## Model Development

Model development adalah tahapan di mana algoritma ML digunakan untuk menjawab problem statement dari tahap business understanding. Pada tahap ini, model ML yang akan diterapkan berjumlah 3 algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Algoritma yang akan kita gunakan, antara lain:

1. Decision Tree
   Decision Tree adalah algoritma machine learning yang menggunakan seperangkat aturan untuk membuat keputusan dengan struktur seperti pohon yang memodelkan kemungkinan hasil, biaya sumber daya, utilitas dan kemungkinan konsekuensi atau resiko. Konsepnya dengan menyajikan algoritma dengan pernyataan bersyarat, meliputi cabang untuk mewakili langkah-langkah pengambilan keputusan yang dapat mengarah pada hasil yang menguntungkan. (Apurb Rajdhan,2020).

```sh
Step penerapan Random Forest (sebelum menggunakan hyperparameter tunning):
    DecisionTreeClassifier(max_depth=6)

Parameter Grid Terbaik :
    criterion: 'gini',max_depth=6,splitter: 'best'
```

2. Random Forest
   Random Forest Random Forest adalah algoritma dalam machine learning yang digunakan untuk pengklasifikasian data set dalam jumlah besar. Karena fungsinya bisa digunakan untuk banyak dimensi dengan berbagai skala dan performa yang tinggi. Klasifikasi ini dilakukan melalui penggabungan tree dalam decision tree dengan cara training dataset yang dimiliki.(Apurb Rajdhan.2020).

```sh
Step penerapan Random Forest (sebelum menggunakan hyperparameter tunning):
    RandomForestClassifier(max_depth=16, n_estimators=50, n_jobs=-1, random_state=55)

Parameter GridSCV Terbaik :
    max_depth=6, n_estimators=50, random_state=55
```

3. AdaBoost
   AdaBoost dikenalkan oleh Freund and Schapire (1995) (Kelleher, John.2020). Awalnya, semua kasus dalam data latih memiliki weight atau bobot yang sama. Pada setiap tahapan, model akan memeriksa apakah observasi yang dilakukan sudah benar? Bobot yang lebih tinggi kemudian diberikan pada model yang salah sehingga mereka akan dimasukkan ke dalam tahapan selanjutnya. Proses iteratif ini berlanjut sampai model mencapai akurasi yang diinginkan.

```sh
Step penerapan AdaBoost (sebelum menggunakan hyperparameter tunning):
    AdaBoostClassifier(learning_rate=0.05, random_state=50)

Parameter GridSCV Terbaik :
    learning_rate=0.1, random_state=5
```

Pada ketiga model di atas diterapkan hyperparameter tuning dengan grid search dan K-Fold cross validation.

1. K-Fold Cross Validation
   Cross-validation adalah teknik yang digunakan untuk evaluasi performa model. Pada CV, data dibagi menjadi K-fold, dimana setiap fold/lipatan digunakan untuk menguji fold/lipatan setidaknya 1 lipatan.
   Cross validation bekerja dengan membagi dataset menjadi grup acak, menjadikan 1 grup untuk grup tes, dan grup sisanya menjadi model latih. Proses ini diulangi tiap grup sampai semua merasakan menjadi grup tes. Lalu, model rata-rata digunakan sebagai hasil akhir model. [[Cross Validation and Grid Search](https://towardsdatascience.com/cross-validation-and-grid-search-efa64b127c1b)].
   K-Fold yang digunakan sebanyak 5, dengan code berikut:
   `kf = StratifiedKFold(n_splits=5, shuffle=False)`

2. Grid Search CV
   Grid Search Cross Validation adalah metode pemilihan kombinasi model dan hyperparameter dengan cara menguji coba satu persatu kombinasi dan melakukan validasi untuk setiap kombinasi. Tujuannya adalah menentukan kombinasi yang menghasilkan performa model terbaik yang dapat dipilih untuk dijadikan model untuk prediksi [11].

## Evaluation

Pada dataset CMS, metrik evaluasi yang digunakan adalah metrik untuk kasus multiclass classification. Metrik yang akan digunakan, yakni **akurasi, precision, recall, F1 score, ROC AUC**.

- Accuracy: menggambarkan seberapa akurat model dalam mengklasifikasikan dengan benar. `Accuracy : (TP + TN)/(TP + TN + FP + FN)`
- Precision: menggambarkan akurasi antara data yang diminta dengan hasil prediksi yang diberikan oleh model. `Precision : (TP) / (TP + FP )`
- Recall / sensitivity: menggambarkan keberhasilan model dalam menemukan kembali sebuah informasi. `Recall : (TP) / (TP + FN)`
- F1-score :menggambarkan perbandingan rata-rata precision dan recall yang dibobotkan. `F1-score : (2 * Precision * Recall) / (Precision + Recall)`
  > Keterangan:
  > True Negative (TN) True Positive (TP)
  > False Negative (FN) False Positive (FP)
- ROC(Receiver Operating Characteristics) AUC (Area Under the Curve)
  Kurva Receiver Operator Characteristic (ROC) adalah metrik evaluasi untuk masalah klasifikasi biner. Ini adalah kurva probabilitas yang memplot TPR terhadap FPR pada berbagai nilai ambang batas dan pada dasarnya memisahkan 'sinyal' dari 'noise'. Dengan kata lain, ini menunjukkan kinerja model klasifikasi di semua ambang batas klasifikasi. Area Di Bawah Kurva (AUC) adalah ukuran kemampuan pengklasifikasi biner untuk membedakan antara kelas dan digunakan sebagai ringkasan kurva ROC. Semakin tinggi AUC, semakin baik performa model dalam membedakan antara kelas positif dan negatif.
  Kurva AUC-ROC hanya untuk masalah klasifikasi biner. Tapi kita bisa memperluasnya ke masalah klasifikasi multikelas menggunakan teknik One vs. All. Jadi, jika kita memiliki tiga kelas, 0, 1, dan 2, ROC untuk kelas 0 akan dihasilkan sebagai mengklasifikasikan 0 melawan bukan 0, yaitu 1 dan 2. ROC untuk kelas 1 akan dihasilkan sebagai mengklasifikasikan 1 melawan bukan 1 , dan seterusnya.

```sh
  Data Prediksi tanpa Grid SCV dan K-Fold Cross Validation
```

---

    Metode         Accuracy    Precision   Recall   F1-Score    ROC AUC
    D.Tree          0.49        0.58        0.49      0.49       0.68
    R.Forest        0.48        0.49        0.48      0.48       0.63
    A.Boost         0.48        0.56        0.48      0.49       0.61

    Data Prediksi dengan Grid SCV dan K-Fold Cross Validation

---

    Metode         Accuracy    Precision   Recall   F1-Score    ROC AUC
    D.Tree          0.58        0.58        0.58      0.58       0.63
    R.Forest        0.53        0.53        0.53      0.51       0.61
    A.Boost         0.56        0.58        0.56      0.55       0.64

```

Dari hasil evaluasi, dapat disimpulkan bahwa:
- performa prediksi terbaik adalah model dari Decision Tree dengan GridCV dan K-Fold Cross Validation, dengan rata-rata akurasi prediksi sebesar 59%.
- fitur yang berpengaruh paling besar adalah `Usia Istri` dan `Pend. Istri`
- User dapat meningkatkan hasil pemakaian kontrasepsi dengan memberi perhatian lebih kepada wanita berusia muda produktif, wanita berpendidikan sekitar SMA ke atas, mengedukasi laki-laki berpendidikan sekitar SMA ke atas, dan menggalakkan media massa sebagai sarana kampanye mengenai metode kontrasepsi.

## Daftar Pustaka
1. Apurb Rajdhan. et al. _Heart Disease Prediction using Machine Learning._ International Journal of Engineering Research & Technology (IJERT). Appl.9, 2278-0181 (2020).
2. Asih, L., Oesman, H., 2009. _Analisa Lanjut SDKI 2007 Faktor yang Memengaruhi Pemakaian Kontrasepsi Jangka Panjang._ Laporan Hasil Penelitian. Jakarta: KB dan Kespro, BKKBN.
3. Bhandari, Aniruddha.2023. [_Panduan untuk Kurva ROC AUC dalam Pembelajaran Mesin: Apa Kekhususannya?_](https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/)
4. Binar Academy.[_Data Pipeline: Pengertian, Proses, dan Jenisnya._](https://www.binaracademy.com/blog/data-pipeline-adalah)
4. Brownlee, Jason.2020._How to Combine Oversampling and Undersampling for Imbalanced Classification._ https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/)
5. Brownie, Jason.2021. [SMOTE for Imbalanced Classification with Python](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
6. Budijanto, 2013. _Determinan “4 Terlalu” Masalah Kesehatan Reproduksi Hubungannya dengan Penggunaan Alat KB Saat Ini di Indonesia._ Buletin Jendela Data dan Informasi Kesehatan. volume II: 17–24.
7. [Contraceptive Method Choice](https://www.kaggle.com/datasets/faizunnabi/contraceptive-method-choice)
8. [Cross Validation and Grid Search](https://towardsdatascience.com/cross-validation-and-grid-search-efa64b127c1b).
9. Dina Elreedy & Amir F. _A Comprehensive Analysis of Synthetic Minority Oversampling Technique (SMOTE) for handling class imbalance._ Information Sciences. Appl. 505, 0020-0255 (2019)
10. Hartayuni Sain & Santi Wulan Purnami._Combine Sampling Support Vector Machine for Imbalanced Data Classification._ Procedia Computer Science. Appl.72, 1877-0509 (2015).
11. Hidayat, Muhammad Ariqleesta.2021. [_GRIDSEARCHCV_](etlify.app/blog/gridsearchcv/)
12. Kelleher, John D, et al._"Machine Learning for Predictive Data Analytics"_.MIT Press.2020
13. Rona A & Vemmie N._Penerapan Kombinasi SMOTE dan Tomek Links untuk Klasifikasi Data Tidak Seimbang dengan Metode Random_
14. Rutecki,Marcin.2023._Kaggle Notebook: SMOTE and Tomek Links for imbalanced data_
15. _Scikit-learn: Machine Learning in Python, Pedregosa et al._, JMLR 12, pp. 2825-2830, 2011.
16. [Seaborn: Visualizing categorical data](https://seaborn.pydata.org/tutorial/categorical.html)
17. Septalia, R., Puspitasari, N., Biostatistika, D., Fakultas, K., Masyarakat, K., Airlangga, U., Mulyorejo Kampus, J., Surabaya, U., &#38; Korespondensi, A. (n.d.). _Faktor yang Memengaruhi Pemilihan Metode Kontrasepsi_
```
