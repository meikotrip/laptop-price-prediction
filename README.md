# Laporan Proyek Machine Learning - Muhamad Meiko Tripurta

## Teknologi, Telekomunikasi

Di era digital, kebutuhan akan laptop semakin meningkat untuk keperluan profesional, pendidikan, dan hiburan. Pasar laptop berkembang dengan berbagai merek dan spesifikasi, membuat konsumen sulit memilih laptop sesuai kebutuhan, terutama terkait harga. Faktor-faktor seperti spesifikasi teknis, fitur yang ditawarkan dan sebagainya sangat mempengaruhi harga laptop. 

Oleh karena itu, kemampuan memprediksi harga laptop berdasarkan spesifikasi tersebut dapat menjadi referensi konsumen untuk membuat keputusan pembelian yang cerdas, serta bagi pengecer dan produsen untuk menetapkan harga yang kompetitif.

Pada awal tahun 2020 saat pandemi Covid-19 melanda seluruh belahan dunia merupakan masa dimana permintaan akan produk laptop sangat tinggi. Permintaan yang tinggi serta pembatasan yang terjadi membuat harga laptop tidak stabil dan cenderung sangat mahal. Hal ini membuat para konsumen kebingungan dalam memutuskan untuk membeli hingga sampai pasca pandemi sekarang. Proyek ini diharapkan dapat menjadi penunjang bagi para konsumen untuk dapat menentukan keputusannya dengan baik dalam memilih laptop.

Referensi : [Keputusan Pembelian Produk Laptop Acer di Era Covid-19](https://journals.upi-yai.ac.id/index.php/IKRAITH-EKONOMIKA/article/download/1705/1406)

## Business Understanding

Dalam industri teknologi yang terus berkembang, laptop telah menjadi perangkat esensial bagi berbagai kalangan, mulai dari pelajar, profesional, hingga pebisnis. Seiring dengan permintaan yang tinggi, pasar laptop menjadi semakin kompetitif dengan banyaknya merek dan model yang tersedia. Setiap model memiliki kombinasi spesifikasi yang berbeda, seperti brand, jenis prosesor, CPU, GPU, tipe RAM, kapasitas RAM, jenis penyimpanan, kapasitas penyimpanan dan fitur-fitur tambahan, yang semuanya mempengaruhi harga akhir laptop.

Masalah utama yang dihadapi oleh konsumen adalah kesulitan dalam membayangkan nilai sebuah laptop berdasarkan spesifikasinya. Mereka sering kali bingung dengan harga yang beragam untuk laptop dengan spesifikasi yang serupa. Di sisi lain, pengecer dan produsen membutuhkan strategi yang efektif untuk menetapkan harga yang kompetitif guna menarik lebih banyak pelanggan dan memaksimalkan keuntungan.

Bagian laporan ini mencakup:

### Problem Statements

Berdasarkan kondisi yang telah diuraikan diatas, Kami mengembangkan sebuah proyek untuk prediksi harga laptop untuk menjawab permasalahan berikut.
- Apa fitur yang paling berpengaruh terhadap harga laptop?
- Berapa harga pasar laptop dengan spesifikasi dan fitur tertentu?

### Goals

Untuk  menjawab pertanyaan tersebut, Anda akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:
- Mengetahui fitur apa yang paling berkorelasi dengan harga laptop
- Membuat model machine learning yang dapat memprediksi harga laptop seakurat mungkin dari fitur-fitur yang ada.

### Solution statements
Solusi yang diberikan yaitu dengan menggunakan 4 algoritma seperti KNN, Random Forest, Linear Regression dan Decision Tree untuk mendapatkan solusi yang terbaik dan menggunakan metrik evaluasi MSE dan R-squared Score untuk mengukur masing-masing model yang dibuat

## Data Understanding
Data yang digunakan pada proyek kali ini adalah [Laptop Price Prediction Dataset](https://www.kaggle.com/datasets/jacksondivakarr/laptop-price-prediction-dataset/code) yang dapat dibuka dan diunduh pada link tersebut. Dataset terdiri dari 893 data dengan total 18 fitur. Dataset ini merupakan kumpulan informasi laptop yang komprehensif yang dirancang untuk mengeksplorasi dan memprediksi harga laptop berdasarkan spesifikasinya.

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- 'Unnamed:0.1' : Index column
- 'Unnamed:0' : Extra column
- 'brand' : Merk dari laptop
- 'name' : Nama dari laptop
- 'price' : Harga eceran laptop (dalam Indian rupee)
- 'spec_rating' : Indikasi rating keseluruhan dari spesifikasi laptop
- 'processor' : Jenis processor laptop yang dipakai
- 'CPU' : Jenis CPU laptop yang digunakan
- 'Ram'  : Kapasistas RAM laptop
- 'Ram_type'  : Tipe RAM laptop
- 'ROM'  : Kapasistas penyimpanan (ROM) laptop
- 'ROM_type'  : Tipe penyimpanan (ROM) laptop
- 'GPU'  : Jenis GPU laptop yang dipakai
- 'display_size'  : Ukuran Layar Laptop
- 'resolution_width'  : Lebar resolusi Laptop
- 'resolution_height'  : Tinggi resolusi Laptop
- 'OS'  : Sistem Operasi Laptop
- 'Warranty'  : jaminan garansi Laptop (tahun)

Beberapa visualisasi yang dilakukan dalam EDA dapat dilihat pada notebook.

## Data Preparation
Beberapa yang dilakukan dalam menyiapkan data untuk model yaitu dengan:

### Encoding fitur kategori pada data menggunakan OneHotEncoder
Encoding fitur adalah proses mengubah data kategorikal (seperti teks atau label) menjadi bentuk numerik yang dapat digunakan oleh algoritma machine learning. Hal ini dilakukan karena sebagian besar algoritma machine learning hanya dapat menangani input numerik, data kategorikal perlu diubah menjadi angka (0 atau 1). 

### Membagi Dataset menjadi training dan testing (split data)
Split data adalah proses membagi dataset menjadi beberapa bagian yang berbeda untuk tujuan pelatihan, validasi, dan pengujian model machine learning. Biasanya, dataset dibagi menjadi dua yaitu training set dan test set. Hali ini dilakukan untuk mengevaluasi model dengan data yang tidak pernah dilihat sebelumnya (data test) selama masa training. Dataset dibagi dengan proporsi 87.5% data traning dan 12.5% data testing.

### Scaling dengan standarisasi nilai pada data
Standarisasi (Standardization) adalah teknik scaling di mana fitur-fitur dalam dataset diubah sehingga memiliki mean (rata-rata) 0 dan standard deviation (simpangan baku) 1. 
Scaling data dengan standarisasi adalah langkah penting dalam proses data preparation yang bertujuan untuk mengubah skala fitur-fitur dalam dataset sehingga memiliki distribusi yang seragam. Selain itu juga dapat mempercepat konvergensi algoritma yang digunakan dalam model prediksi.

## Modeling
Pada tahap modeling, terdapat beberapa model Machine Learning yang digunakan yaitu K Nearest Neighbor (KNN), Random Forest, Linear Regression, dan Decision Tree. 

### K Nearest Neighbor
tahapan awal dalam membangun model machine learning dengan algoritma KNN yaitu  dengan mengimport KNeighborsRegressor dari library sklearn lalu diisi dengan parameter n_neighbors. setelah di inisiasi selanjutnya model ditraining dengan dataset train.

Kelebihan KNN :
- Sederhana dan mudah dipahami
- Fleksibel

Kekurangan KNN :
- Lambat mengolah data dalam jumlah besar
- Sensitif terhadap skala fitur

### Random Forest
tahapan awal dalam membangun model machine learning dengan algoritma Random Forest yaitu dengan mengimport RandomForestRegressor dari library sklearn lalu diisi dengan parameter n_estimators, max_depth, random_state, dan n_jobs. setelah di inisiasi selanjutnya model ditraining dengan dataset train.

Kelebihan Random Forest :
- memberikan akurasi yang tinggi
- robust terhadap noise

Kekurangan Random Forest :
- model menjadi sangat besar dan kompleks
- waktu komputasi yang tinggi

### Linear Regression
tahapan awal dalam membangun model machine learning dengan algoritma Linear Regression yaitu dengan mengimport LinearRegression dari library sklearn. setelah di inisiasi selanjutnya model ditraining dengan dataset train.

Kelebihan Linear Regression :
- sederhana dan mudah dipahami
- waktu pelatihan dan prediksi yang sangat cepat

Kekurangan Linear Regression :
- sensitif terhadap outliers
- kurang baik untuk fitur interaksi

### Decision Tree
tahapan awal dalam membangun model machine learning dengan algoritma Decision Tree yaitu  dengan mengimport DecisionTreeRegressor dari library sklearn lalu diisi dengan parameter max_depth dan random_state. setelah di inisiasi selanjutnya model ditraining dengan dataset train.

Kelebihan Decision Tree :
- mudah dipahami dan diintepretasikan
- tidak membutuhkan feature scaling

Kekurangan Decision Tree :
- tidak robust terhadap noise
- cenderung overfitting

Dari keempat model machine learning yang dibuat, model machine learning prediksi harga laptop terbaik yaitu terdapat pada Model dengan algoritma Random Forest karena metriks evaluasi MSE dan R2_Score nya lebih baik dibandingkan algoritma yang lain serta prediksi harganya yang mendekati dengan harga sebenarnya pada data test

## Evaluation
Pada tahap evaluasi, beberapa metriks evaluasi yang digunakan untuk mengevaluasi model-model diatas yaitu dengan mean squared error (MSE) dan R-squared score.

### Mean Squared Error (MSE)
Mean Squared Error (MSE) adalah metrik yang mengukur rata-rata kuadrat perbedaan antara nilai yang diprediksi oleh model dan nilai aktual dari data. MSE memberikan gambaran tentang seberapa besar kesalahan model dalam hal nilai numerik.

Rumus :

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- yi adalah nilai aktual
- yi_hat adalah nilai yang diprediksi model
- n adalah jumlah data

### R-squared Score
R-squared (R²) Score adalah metrik yang mengukur seberapa baik model regresi dapat menjelaskan variabilitas data target. Ini adalah ukuran proporsi variansi dalam data target yang dapat dijelaskan oleh model.

Rumus :

\[
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
\]

- SS_res adalah jumlah kuadrat dari residual (perbedaan antara nilai aktual dan nilai prediksi).
- SS_tot adalah jumlah kuadrat dari deviasi nilai aktual dari rata-rata nilai aktual.

Dari hasil prediksi dan hasil evaluasi model dengan melihat metriks-metriks yang digunakan didapatkan bahwa algoritma dengan hasil prediksi yang paling mendekati yaitu algoritma <b>Random Forest</b> (RF) dengan test_mse(310) dan test_r2(0.74)
