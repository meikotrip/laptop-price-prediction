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

Beberapa insight yang diperoleh dari hasil exploratory data analysis yaitu sebagai berikut:

- Terdapat 29 kategori brand dengan brand terbanyak yaitu HP sebanyak 170 laptop
- Terdapat 165 kategori processor dengan processor terbanyak yaitu 12th Gen Intel Core i5 1235U sebanyak 49 laptop
- Terdapat 24 kategori CPU dengan CPU terbanyak yaitu Quad Core, 8 Threads sebanyak 130 laptop
- Terdapat 12 kategori tipe RAM dengan tipe RAM terbanyak yaitu DDR4 sebanyak 497 laptop
- Terdapat 2 kategori tipe ROM dengen tipe ROM terbanyak yaitu SSD sebanyak 801 laptop
- Terdapat 115 kategori GPU dengan GPU terbanyak yaitu Intel Iris Xe Graphics sebanyak 98 laptop
- Terdapat 12 kategori OS dengan OS terbanyak yaitu Windows 11 OS sebanyak 723 laptop
- Rentang harga laptop cukup tinggi yaitu dari skala 10000 hingga sekitar 160000.
- Harga laptop lebih banyak berada di sekitar kurang dari 100000.
- Distribusi harga miring ke kanan (right-skewed).

## Data Preparation
Beberapa yang dilakukan dalam menyiapkan data untuk model yaitu dengan:

### Encoding fitur kategori pada data menggunakan OneHotEncoder
Encoding fitur adalah proses mengubah data kategorikal (seperti teks atau label) menjadi bentuk numerik yang dapat digunakan oleh algoritma machine learning. Hal ini dilakukan karena sebagian besar algoritma machine learning hanya dapat menangani input numerik, data kategorikal perlu diubah menjadi angka (0 atau 1). Feature yang diencode pada data tersebut yaitu feature 'ROM_type' 

### Membagi Dataset menjadi training dan testing (split data)
Split data adalah proses membagi dataset menjadi beberapa bagian yang berbeda untuk tujuan pelatihan, validasi, dan pengujian model machine learning. Biasanya, dataset dibagi menjadi dua yaitu training set dan test set. Hali ini dilakukan untuk mengevaluasi model dengan data yang tidak pernah dilihat sebelumnya (data test) selama masa training. Dataset dibagi dengan proporsi 87.5% data traning dan 12.5% data testing.

### Scaling dengan standarisasi nilai pada data
Standarisasi (Standardization) adalah teknik scaling di mana fitur-fitur dalam dataset diubah sehingga memiliki mean (rata-rata) 0 dan standard deviation (simpangan baku) 1. 
Scaling data dengan standarisasi adalah langkah penting dalam proses data preparation yang bertujuan untuk mengubah skala fitur-fitur dalam dataset sehingga memiliki distribusi yang seragam. Selain itu juga dapat mempercepat konvergensi algoritma yang digunakan dalam model prediksi. Feature yang dilakukan strandarisasi yaitu feature numeric seperti 'spec_rating', 'Ram', 'ROM', 'resolution_width', 'resolution_height'

## Modeling
Pada tahap modeling, terdapat beberapa model Machine Learning yang digunakan yaitu K Nearest Neighbor (KNN), Random Forest, Linear Regression, dan Decision Tree. 

### K Nearest Neighbors
Algoritma KNN (K-Nearest Neighbors) bekerja dengan cara mengidentifikasi k tetangga terdekat dari sebuah data berdasarkan metrik jarak tertentu (biasanya jarak Euclidean), kemudian menggunakan rata-rata nilai target dari tetangga-tetangga tersebut untuk membuat prediksi.

Langkah-langkah yang dilakukan:
1. Mengimpor library 'KNeighborsRegressor' dari library 'sklearn.neighbors'
2. Inisialisasi model KNN dengan parameter 'n_neighbors = 6'
3. Training Model dengan data train agar model dapat memahami pola dari data

Kelebihan KNN :
- Sederhana dan mudah dipahami
- Fleksibel

Kekurangan KNN :
- Lambat mengolah data dalam jumlah besar
- Sensitif terhadap skala fitur

### Random Forest
Algoritma Random Forest bekerja dengan cara membangun banyak pohon keputusan (decision trees) selama pelatihan dan menghasilkan prediksi rata-rata dari setiap pohon untuk meningkatkan akurasi dan mengurangi overfitting.

Langkah-langkah yang dilakukan:
1. Mengimpor library 'RandomForestRegressor' dari library 'sklearn.ensemble'
2. Inisialisasi model dengan parameter 'n_estimators = 100', 'max_depth = 32', 'random_state = 64', 'n_jobs = 1'
3. Training Model dengan data train agar model dapat memahami pola dari data

Kelebihan Random Forest :
- memberikan akurasi yang tinggi
- robust terhadap noise

Kekurangan Random Forest :
- model menjadi sangat besar dan kompleks
- waktu komputasi yang tinggi

### Linear Regression
Linear Regression adalah salah satu algoritma paling dasar dalam machine learning yang digunakan untuk memodelkan hubungan antara satu atau lebih fitur independen (predictors) dan variabel target kontinu. Algoritma ini bekerja dengan mencari garis linear (atau hyperplane dalam kasus multivariat) yang paling sesuai dengan data, meminimalkan kesalahan antara prediksi dan nilai aktual.

Langkah-langkah yang dilakukan:
1. Mengimpor library 'LinearRegression' dari library 'sklearn.linear_model'
2. Inisialisasi model LinearRegression tanpa parameter
3. Training Model dengan data train agar model dapat memahami pola dari data

Kelebihan Linear Regression :
- sederhana dan mudah dipahami
- waktu pelatihan dan prediksi yang sangat cepat

Kekurangan Linear Regression :
- sensitif terhadap outliers
- kurang baik untuk fitur interaksi

### Decision Tree
Decision Tree adalah salah satu algoritma machine learning yang digunakan untuk melakukan prediksi dengan cara memetakan pengambilan keputusan ke dalam struktur pohon. Algoritma ini membagi data berdasarkan fitur-fitur tertentu dan membuat keputusan akhir dengan mengikuti cabang-cabang pohon yang terbentuk. Pada model regresi, decision tree memprediksi nilai target dengan menghitung rata-rata nilai dari data yang berada di leaf node.

Langkah-langkah yang dilakukan:
1. Mengimpor library 'DecisionTreeRegressor' dari library 'sklearn.tree'
2. Inisialisasi model dengan parameter 'max_depth = 32', 'random_state = 64'
3. Training Model dengan data train agar model dapat memahami pola dari data

Kelebihan Decision Tree :
- mudah dipahami dan diintepretasikan
- tidak membutuhkan feature scaling

Kekurangan Decision Tree :
- tidak robust terhadap noise
- cenderung overfitting

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

$$
\text{R²} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
$$

- SS_res adalah jumlah kuadrat dari residual (perbedaan antara nilai aktual dan nilai prediksi).
- SS_tot adalah jumlah kuadrat dari deviasi nilai aktual dari rata-rata nilai aktual.

Sebagai kesimpulan akhir dari analisis yang telah kita lakukan dari awal dapat ditarik beberapa poin bahwasannya :
- Dalam analisis yang dilakukan tidak secara eksplisit memberikan fitur apa yang paling berpengaruh terhadap harga laptop, namun feature yang memiliki korelasi paling tinggi yaitu pada fitur 'spec_rating' dengan skor 0.48. Analisis juga terus dikembangkan hingga pembuatan model machine learning untuk prediksi harga barang dari fitur-fitur yang relevan
- Dari hasil prediksi dan hasil evaluasi 4 model yang dibuat (KNN, Random Forest, Linear Regression, dan Decision Tree) daro metriks-metriks yang digunakan didapatkan bahwa algoritma dengan hasil prediksi yang paling mendekati yaitu algoritma <b>Random Forest</b> (RF) dengan test_mse(310) dan test_r2(0.74)
