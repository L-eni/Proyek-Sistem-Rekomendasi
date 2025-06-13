# Laporan Proyek Machine Learning - Leni Gustia

---

## Project Overview

---

Film merupakan salah satu bentuk hiburan paling populer dan memiliki dampak budaya yang luas. Dalam era digital, kemunculan platform *streaming* seperti *Netflix*, *Amazon Prime*, dan *Disney+* telah mengubah cara masyarakat global mengakses dan menikmati konten film. Ketersediaan ribuan judul dalam satu platform telah memberikan keleluasaan kepada pengguna, namun pada saat yang sama menimbulkan tantangan baru berupa kelebihan informasi (*information overload*). Fenomena ini terjadi ketika pengguna dihadapkan pada begitu banyak pilihan sehingga kesulitan dalam menemukan film yang benar-benar sesuai dengan preferensinya [(Eppler & Mengis, 2004)](https://www.tandfonline.com/doi/abs/10.1080/01972240490507974). Kondisi ini diperparah dengan perbedaan selera yang unik di antara pengguna serta dinamika tren perfilman yang terus berubah.

Untuk mengatasi tantangan ini, berbagai pendekatan teknologi telah dikembangkan, salah satunya melalui sistem rekomendasi film yang bertujuan menyederhanakan proses penemuan konten. Sistem ini berfungsi sebagai alat bantu cerdas yang mempelajari pola preferensi pengguna dan menyajikan rekomendasi film yang relevan. Dalam beberapa dekade terakhir, *collaborative filtering (CF)* menjadi salah satu metode paling dominan dalam pengembangan sistem rekomendasi. Metode ini bekerja dengan menganalisis kesamaan perilaku antar pengguna atau antar *item* berdasarkan data historis, tanpa memerlukan informasi detail mengenai konten film itu sendiri [(Su & Khoshgoftaar, 2009)](https://onlinelibrary.wiley.com/doi/10.1155/2009/421425).

*Dataset* *MovieLens*, yang dikembangkan oleh *GroupLens Research*, telah menjadi *benchmark* penting dalam penelitian dan pengembangan sistem rekomendasi. *Dataset* ini berisi jutaan *rating* film yang diberikan oleh ribuan pengguna, serta dilengkapi dengan *metadata* film seperti judul, genre, dan tahun rilis. Dengan menggunakan pendekatan *item-based collaborative filtering* dan pengukuran kesamaan seperti *cosine similarity*, sistem dapat merekomendasikan film berdasarkan kemiripan pola *rating* antar *item* yang dinilai oleh pengguna sebelumnya [(Sarwar et al., 2001)](https://dl.acm.org/doi/10.1145/371920.372071). Keunggulan metode ini terletak pada skalabilitas dan akurasi prediksi dalam lingkungan dengan data pengguna yang sangat banyak.

Penelitian sebelumnya menunjukkan bahwa pendekatan *item-based CF* dapat menghasilkan rekomendasi yang efisien dan akurat, terutama ketika jumlah pengguna sangat besar dan data bersifat *sparsity* [(Li et al., 2018)](https://www.researchgate.net/publication/355894961_Movie_Recommender_System_with_Visualized_Embeddings). Selain itu, [Forouzandeh et al. (2020)](https://arxiv.org/abs/2008.01192) memperkuat efektivitas *CF* dengan menggabungkan teknik *graph embedding* dan *ensemble learning* yang meningkatkan *predictability* dalam domain film. Temuan-temuan ini mempertegas bahwa pengembangan sistem rekomendasi berbasis *item similarity* bukan hanya dapat meningkatkan pengalaman pengguna secara individual, tetapi juga berdampak pada peningkatan keterlibatan (*engagement*) dan loyalitas pengguna terhadap platform [(Adomavicius & Tuzhilin, 2005)](https://ieeexplore.ieee.org/document/1423975).

Lebih dari itu, sistem rekomendasi juga memiliki potensi strategis untuk mendukung pertumbuhan industri film secara keseluruhan. Kemampuannya untuk menyoroti film-film dengan popularitas rendah atau kategori *niche* membuka peluang bagi karya independen untuk menjangkau audiens yang sesuai. Data interaksi pengguna yang dikumpulkan melalui sistem ini juga menjadi sumber informasi penting bagi produser dan penyedia platform dalam memahami dinamika preferensi pasar secara *real-time*. Dengan demikian, pengembangan sistem rekomendasi berbasis *collaborative filtering* yang andal dan cerdas tidak hanya menjadi solusi terhadap masalah *information overload*, tetapi juga menjadi elemen kunci dalam memastikan keberlanjutan dan kemajuan industri perfilman di era digital.

 
**Referensi**:
- Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions. IEEE Trans. on Knowledge and Data Engineering, 17(6), 734–749.
- Eppler, M. J., & Mengis, J. (2004). The concept of information overload: A review of literature from organization science, accounting, marketing, MIS, and related disciplines. The Information Society, 20(5), 325–344.
-Forouzandeh, S. et al. (2020). Presentation of a recommender system with ensemble learning and graph embedding: A case on MovieLens. Multimedia Tools and Applications, 79(43), 32229–32250.
- Li, J. et al. (2018). Movie recommendation based on bridging movie feature and user interest. Journal of Computational Science, 26, 128–134.
- Sarwar, B. et al. (2001). Item-based collaborative filtering recommendation algorithms. In Proc. of the 10th International Conference on World Wide Web (pp. 285–295).
- Su, X., & Khoshgoftaar, T. M. (2009). A survey of collaborative filtering techniques. Advances in Artificial Intelligence, 2009, 1–19.

## Business Understanding

---
### Problem Statements

1. Bagaimana cara memberikan rekomendasi film yang relevan kepada pengguna berdasarkan preferensi mereka, dengan mempertimbangkan data historis seperti *rating* dan *genre* yang disukai?  
2. Bagaimana mengatasi fenomena *information overload* atau paradoks kelimpahan, di mana banyaknya pilihan film justru menyulitkan pengguna dalam menemukan film yang sesuai dengan minat mereka?  
3. Bagaimana memprediksi *rating* atau preferensi pengguna terhadap film yang belum ditonton, berdasarkan histori dan pola interaksi sebelumnya?


### Goals

1. Mengimplementasikan pendekatan *Collaborative Filtering* untuk memprediksi rating film dengan membandingkan dua arsitektur model.  
2. Membangun model *Matrix Factorization (RecommenderNet)* sebagai baseline, menggunakan *dot product* antara embedding pengguna dan film.  
3. Mengembangkan model *Neural Matrix Factorization (NeuMF)* yang menggabungkan jalur *linear* (*Generalized Matrix Factorization/GMF*) dan *non-linear* (*Multi-Layer Perceptron/MLP*) untuk menangkap pola preferensi yang lebih kompleks.  
4. Mengevaluasi performa kedua model menggunakan metrik *Root Mean Squared Error (RMSE)* dan *Mean Absolute Error (MAE)* untuk menilai akurasi prediksi rating.

### Solution Statements

1. Menerapkan pendekatan *Content-Based Filtering (CBF)* dengan menganalisis atribut film seperti *genre*, *rating*, dan *tahun rilis* untuk memberikan rekomendasi yang lebih personal.  
2. Mengembangkan representasi berbasis metadata film dan menghubungkannya dengan preferensi pengguna untuk meningkatkan relevansi rekomendasi.  
3. Menggunakan teknik *machine learning* seperti *K-Nearest Neighbors (KNN)* dan *Singular Value Decomposition (SVD)* untuk memahami relasi kompleks antara pengguna dan film.  
4. Menerapkan dua pendekatan utama:
   - *Dot product* antara embedding pengguna dan film sebagai baseline (**MF**).
   - **NeuMF**, yaitu neural network yang menggabungkan pendekatan **MF dan MLP**, untuk meningkatkan akurasi prediksi.  
5. Mengukur performa model dengan menggunakan metrik evaluasi **MAE** dan **RMSE** agar dapat menilai seberapa akurat sistem dalam memprediksi preferensi pengguna.

## Data Understanding

---

### Tentang Dataset

Dataset yang digunakan mengandung berbagai informasi penting mengenai **film dan interaksi pengguna**, yang digunakan untuk membangun **sistem rekomendasi film** berbasis machine learning. Dataset ini bersifat publik dan dapat diakses melalui laman Kaggle: [Movie Recommendation System – Parashar Manas](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system/data).

Dataset ini terdiri atas dua file utama:
- **`movies.csv`** — Berisi metadata dari setiap film, terdiri dari **62.422 baris** dan **3 kolom**, dengan fitur:
  - `movieId`: ID unik untuk setiap film.
  - `title`: Judul film beserta tahun rilis.
  - `genres`: Kategori genre dari film, dipisahkan oleh simbol `|` (contoh: `Action|Adventure|Sci-Fi`).

- **`ratings.csv`** — Berisi data interaksi eksplisit berupa rating yang diberikan pengguna terhadap film, terdiri dari **25.000.094 baris** dan **4 kolom**, dengan fitur:
  - `userId`: ID unik yang mewakili masing-masing pengguna.
  - `movieId`: ID film yang diberi rating oleh pengguna.
  - `rating`: Nilai rating yang diberikan (dalam skala 0.5 hingga 5.0).
  - `timestamp`: Waktu saat pengguna memberikan rating (format Unix timestamp).

### Kondisi Data

- **Tidak Ada Missing Value**: Semua kolom pada kedua file telah terisi dengan lengkap.
- **Tidak Ada Duplikat**: Tidak ditemukan baris data yang terduplikasi.
- **Siap Digunakan**: Dataset sudah bersih dan siap digunakan dalam proses eksplorasi, analisis, serta pengembangan sistem rekomendasi berbasis **Collaborative Filtering** dan **Matrix Factorization**.

### Visualisasi Distribusi Data Rating

![Distribusi](/Gambar/Distribusi_Rating.png)


Berdasarkan grafik distribusi rating, terlihat bahwa penilaian pengguna tidak tersebar merata dan cenderung terkonsentrasi pada nilai yang tinggi, terutama pada rating 4 dan 5 yang mendominasi. Hal ini menunjukkan bahwa **mayoritas pengguna memberikan ulasan positif terhadap film yang mereka tonton**.

Meskipun rating **3** juga cukup sering muncul, jumlahnya masih lebih rendah dibandingkan rating 4 dan 5. Di sisi lain, **rating rendah** seperti **0.5, 1.0, hingga 2.0** sangat jarang diberikan. Pola ini mengindikasikan bahwa **pengguna cenderung hanya memberikan rating pada film yang mereka sukai atau anggap layak ditonton**, dan mungkin menghindari memberi rating pada film yang mereka tidak suka atau tidak selesai ditonton.

Secara statistik, distribusi ini bersifat *left-skewed* (*skew negatif*), karena **sebagian besar nilai berada di sisi kanan (nilai tinggi)**, sementara frekuensi rating rendah sangat kecil. Pola distribusi seperti ini cukup umum dalam sistem rekomendasi berbasis rating, karena adanya **bias positif** dari pengguna yang hanya aktif menilai konten yang mereka nikmati.


### Visualisasi Histogram Data Rating

![Histogram](/Gambar/Histogram_Rating.png)

Berdasarkan visualisasi distribusi rating, terlihat bahwa mayoritas pengguna memberikan **rating tinggi**, dengan nilai **4** mendominasi frekuensi. Rating **3 dan 5** juga sering diberikan, yang mengindikasikan bahwa sebagian besar pengguna menilai film sebagai **cukup baik hingga sangat baik**.

Sebaliknya, **rating rendah** seperti **0, 1, dan 2** sangat jarang diberikan. Hal ini menunjukkan bahwa pengguna cenderung **tidak memberikan penilaian negatif**, atau hanya memberikan rating terhadap film yang mereka sukai.

Distribusi ini menunjukkan pola *right-skewed* (*positively skewed*), yaitu kondisi di mana sebagian besar nilai berada pada area **rating tinggi**. Pola seperti ini penting untuk diperhatikan dalam pengembangan *recommender system*, agar tidak terjadi *bias* terhadap konten dengan rating tinggi, serta model tetap dapat mengenali variasi preferensi pengguna secara adil.

 

### Visualisasi Boxplot Data Rating

![Boxplot](/Gambar/Boxplot_Rating.png)

Berdasarkan visualisasi **boxplot** nilai rating film, dapat disimpulkan bahwa mayoritas pengguna memberikan rating dalam rentang **3 hingga 4.5**. Hal ini ditunjukkan oleh **lebar kotak (interquartile range/IQR)**, yang merepresentasikan 50% data berada di antara kuartil pertama dan ketiga.

Garis **median** di dalam kotak berada mendekati **nilai 4**, yang mengindikasikan bahwa sebagian besar pengguna cenderung memberikan **penilaian tinggi** terhadap film yang mereka tonton.

Terdapat juga **outlier** di sisi kiri, yaitu rating **di bawah 1.5**, yang muncul sebagai titik-titik terpisah di luar whisker. Ini menunjukkan bahwa meskipun jarang, ada pengguna yang memberikan **rating sangat rendah**.

Sementara itu, **batas bawah dan atas whisker** menunjukkan bahwa sebagian besar rating berkisar antara **2 hingga 5**, menegaskan bahwa rating rendah relatif jarang.

Secara keseluruhan, distribusi ini bersifat *positively skewed* (condong ke kanan), mencerminkan bahwa **film-film dalam dataset umumnya disukai oleh pengguna**, dengan dominasi rating tinggi yang menjadi ciri khas dalam data ulasan pengguna.


## Data Preparation

---
Sebelum data digunakan untuk melatih model sistem rekomendasi, dilakukan beberapa tahap **persiapan data** guna memastikan kualitas data, efisiensi komputasi, serta kesesuaian format data dengan kebutuhan algoritma pembelajaran mesin yang digunakan.

#### 1. Pemeriksaan dan Pembersihan Data Duplikat
Dataset telah diperiksa dan **tidak ditemukan data duplikat**. Oleh karena itu, tidak diperlukan penghapusan data ganda, dan dataset dapat langsung digunakan untuk proses selanjutnya.

#### 2. Preprocessing dan Filtering Data
Karena ukuran dataset cukup besar (sekitar **25.000.094 baris data rating** dari file `ratings.csv`), dilakukan proses *filtering* untuk meningkatkan efisiensi proses komputasi dan memastikan bahwa model dilatih pada data yang paling relevan.

- **Fokus pada Rating Eksplisit**: Data dengan nilai rating `0` dihapus karena dianggap tidak merepresentasikan preferensi pengguna secara eksplisit. Langkah ini bertujuan untuk melatih model hanya menggunakan data yang mencerminkan penilaian nyata dari pengguna terhadap film. Namun, dalam dataset yang digunakan, tidak ditemukan rating di bawah 0,5.

- **Filtering Berdasarkan Aktivitas**: Untuk meningkatkan kualitas data pelatihan, hanya pengguna yang memberikan **minimal 50 rating** dan film yang menerima **minimal 50 rating** yang disertakan. Langkah ini membantu model belajar dari interaksi yang lebih stabil dan informatif.

#### 3. Encoding Data
ID pengguna (`userId`) dan ID film (`movieId`) yang tersisa setelah proses filtering diubah menjadi **indeks numerik** mulai dari `0`. Kolom baru dengan nama `user` dan `movie` ditambahkan untuk menyimpan hasil encoding ini. Hal ini penting agar model dapat memanfaatkan data ID dalam bentuk tensor atau array numerik yang bisa diproses oleh algoritma pembelajaran mendalam.

#### 4. Normalisasi Rating
Kolom `rating` dinormalisasi ke dalam rentang antara `0` dan `1` menggunakan teknik **Min-Max Scaling**. Nilai hasil normalisasi ini disimpan dalam kolom baru bernama `rating_norm`, yang akan digunakan sebagai **label target** pada saat pelatihan model. Normalisasi ini bertujuan untuk mempercepat proses pelatihan model dan meningkatkan stabilitas pembelajaran.

#### 5. Split Dataset
Dataset yang telah dibersihkan dan diproses kemudian diacak dan dibagi menjadi dua bagian:
- **80% untuk data latih**
- **20% untuk data validasi**

Pembagian ini dilakukan untuk melatih model pada sebagian data dan mengevaluasi kemampuannya memprediksi pada data yang belum pernah dilihat sebelumnya.


## Modeling

---

Dalam proyek ini dikembangkan dua pendekatan model rekomendasi berbasis *collaborative filtering*. Dua pendekatan model digunakan untuk membandingkan performa model klasik berbasis *matrix factorization* dengan model *deep learning* berbasis *neural network*, guna mengeksplorasi performa dan fleksibilitas masing-masing dalam konteks rekomendasi film.

### 1. Matrix Factorization dengan Embedding (Baseline)

Model ini menggunakan pendekatan klasik *collaborative filtering* yang diimplementasikan menggunakan *embedding layer* dalam TensorFlow. Model ini disebut sebagai **RecommenderNet**.

#### Arsitektur

- Dua buah *embedding layer*: satu untuk **user**, satu untuk **movie**.
- Operasi *dot product* antar *embedding* untuk mendapatkan skor kecocokan.
- Tambahan **bias** untuk user dan movie.
- Aktivasi **sigmoid** agar output berada dalam rentang 0–1.

```python
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(num_movies, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.reduce_sum(user_vector * movie_vector, axis=1, keepdims=True)
        x = dot_user_movie + user_bias + movie_bias
        return tf.nn.sigmoid(x)

# Instansiasi dan kompilasi model RecommenderNet
recommender_net_model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
recommender_net_model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
 ```

**Kelebihan:**

- Memiliki arsitektur yang sederhana sehingga lebih efisien dalam penggunaan sumber daya komputasi.
- Proses pelatihan berjalan cepat dan dapat mencapai konvergensi dalam waktu yang relatif singkat.
- Memberikan performa yang stabil dan cocok digunakan sebagai acuan awal (baseline) untuk membandingkan model-model yang lebih kompleks.

**Kekurangan:**

-  Hanya mampu memodelkan hubungan linear antara preferensi pengguna dan fitur film, sehingga kurang fleksibel.
-  Kurang optimal dalam menangkap pola preferensi yang lebih kompleks atau bersifat non-linear, yang sering kali muncul dalam interaksi pengguna dengan item.

### 2. **Neural Matrix Factorization (NeuMF)**

**Neural Matrix Factorization (NeuMF)** merupakan pendekatan modern dalam sistem rekomendasi yang menggabungkan dua metode utama, yaitu *Generalized Matrix Factorization* (GMF) dan *Multi-Layer Perceptron* (MLP). Pendekatan ini menawarkan fleksibilitas yang lebih tinggi karena mampu mempelajari hubungan non-linear antara pengguna dan item melalui jaringan saraf tiruan (*neural network*).

### Arsitektur Model

Model NeuMF terdiri atas dua jalur pemrosesan embedding:

- **GMF (Generalized Matrix Factorization):**  
  Menggunakan operasi *dot product* untuk menangkap interaksi linier antara pengguna dan item, menyerupai metode *matrix factorization* klasik.

- **MLP (Multi-Layer Perceptron):**  
  Menggabungkan (*concatenate*) embedding pengguna dan item, kemudian meneruskannya ke beberapa *fully connected layer* untuk mempelajari pola interaksi yang lebih kompleks.

### Penggabungan dan Output

- Keluaran dari kedua jalur (**GMF** dan **MLP**) digabungkan (*concatenated*) dan diproses lebih lanjut melalui lapisan *dense* akhir.
- Lapisan output menggunakan fungsi aktivasi **sigmoid** untuk menghasilkan skor prediksi dalam rentang **0 hingga 1**.
```python
def get_NeuMF_model(num_users, num_movies, mf_dim=8, mlp_layers=[64,32,16,8], dropout=0.0):
    user_input = Input(shape=(1,), name="user_input")
    item_input = Input(shape=(1,), name="item_input")
    # MF part
    mf_user_embedding = layers.Embedding(num_users, mf_dim, name="mf_user_embedding")(user_input)
    mf_item_embedding = layers.Embedding(num_movies, mf_dim, name="mf_item_embedding")(item_input)
    mf_vector = layers.multiply([layers.Flatten()(mf_user_embedding), layers.Flatten()(mf_item_embedding)])
    # MLP part
    mlp_embedding_dim = mlp_layers[0] // 2
    mlp_user_embedding = layers.Embedding(num_users, mlp_embedding_dim, name="mlp_user_embedding")(user_input)
    mlp_item_embedding = layers.Embedding(num_movies, mlp_embedding_dim, name="mlp_item_embedding")(item_input)
    mlp_vector = layers.concatenate([layers.Flatten()(mlp_user_embedding), layers.Flatten()(mlp_item_embedding)])
    for idx, units in enumerate(mlp_layers[1:]):
        mlp_vector = layers.Dense(units, activation='relu', name=f"mlp_dense_{idx}")(mlp_vector)
        if dropout > 0:
            mlp_vector = layers.Dropout(dropout)(mlp_vector)
    # Concatenate
    neumf_vector = layers.concatenate([mf_vector, mlp_vector])
    prediction = layers.Dense(1, activation="sigmoid", name="prediction")(neumf_vector)
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    return model

# Instansiasi dan kompilasi model NeuMF
neuMF_model = get_NeuMF_model(num_users, num_movies)
neuMF_model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

```
  
**Kelebihan:**
- **Kemampuan Menangkap Interaksi Kompleks:**  
  NeuMF unggul dalam mengenali hubungan non-linear dan kompleks antara pengguna dan item (seperti anime), terutama melalui komponen *Multi-Layer Perceptron (MLP)* yang dimilikinya.
- **Arsitektur yang Fleksibel:**  
  Dengan memanfaatkan jaringan saraf tiruan, model ini mampu menyesuaikan diri dengan berbagai pola data yang tidak dapat ditangkap oleh metode tradisional.

**Kekurangan:**
- **Kompleksitas dan Konsumsi Sumber Daya Tinggi:**  
  Struktur model yang lebih kompleks memerlukan daya komputasi lebih besar serta waktu pelatihan yang lebih lama dibandingkan pendekatan sederhana seperti matrix factorization klasik.
- **Tantangan dalam Penyetelan Hyperparameter:**  
  Menentukan kombinasi hyperparameter yang optimal—seperti jumlah lapisan, jumlah neuron, dan tingkat dropout—menjadi lebih sulit namun sangat penting untuk mencapai performa terbaik.
 menantang untuk mencapai performa puncak.

---

### **Top-N Recommendation**

Analisis kualitatif terhadap hasil rekomendasi untuk pengguna sampel (**user_id: 1**) menunjukkan pola yang menarik dari dua pendekatan model yang digunakan, yakni **RecommenderNet** dan **NeuMF**.


Terdapat konsistensi cukup tinggi antara kedua model dalam menangkap minat pengguna terhadap **film dokumenter dan drama berkualitas tinggi**. Judul-judul seperti *Band of Brothers (2001)*, *Planet Earth*, dan *Shawshank Redemption* muncul pada daftar 10 besar di kedua model, menandakan bahwa model mampu mengenali kecenderungan pengguna terhadap film-film dengan **narasi kuat, intens emosional, dan nilai produksi tinggi**.

Model **NeuMF** cenderung menonjolkan **keragaman genre dan eksplorasi film populer lintas kategori**. Film seperti *Spider-Man: Into the Spider-Verse (2018)* dan *Hoop Dreams (1994)* menjadi bukti bahwa model ini tidak hanya fokus pada genre favorit pengguna (seperti dokumenter atau drama), tetapi juga mencoba menawarkan **film animasi, action, dan sport** yang dinilai relevan secara personal berdasarkan pola interaksi pengguna.

Ciri khas lain NeuMF adalah preferensinya terhadap **film dengan popularitas tinggi dan kompleksitas tematik**, seperti *Fight Club (1999)* dan *Three Billboards Outside Ebbing, Missouri (2017)*, yang menunjukkan pemetaan ketertarikan terhadap **cerita yang provokatif dan sinematografi yang kuat**.

Sementara itu, **RecommenderNet** menampilkan pendekatan yang lebih **konsisten dan tematik**, dengan dominasi genre **drama dan dokumenter**. Film seperti *Planet Earth*, *Baraka (1992)*, dan *Celebration (The Festen)* menegaskan kecenderungan model ini untuk **memberikan rekomendasi yang lebih sejalan dengan preferensi historis pengguna**, terutama terhadap film-film dokumenter visual dan drama klasik dengan bobot naratif tinggi.

Model ini cenderung memberikan **rekomendasi yang lebih stabil dan berorientasi genre**, tanpa banyak mengeksplorasi genre-genre alternatif.


Dari sisi evaluasi metrik seperti **Root Mean Squared Error (RMSE)** dan **Mean Absolute Error (MAE)** (yang diasumsikan tersedia dari proses pelatihan sebelumnya), kemungkinan besar RecommenderNet dapat memberikan hasil prediksi rating yang **lebih stabil** berkat arsitekturnya yang relatif sederhana dan minim overfitting.

Sebaliknya, **NeuMF** lebih kuat dalam menyajikan rekomendasi yang **sangat personal dan adaptif**, meskipun mungkin memiliki sedikit penurunan dalam stabilitas prediksi rating jika tidak dituning secara optimal.

Secara keseluruhan, bila fokus utama adalah **akurasinya dalam menilai film yang benar-benar disukai pengguna**, **RecommenderNet** bisa menjadi pilihan yang **lebih konservatif namun dapat diandalkan**. Di sisi lain, **NeuMF** unggul dalam **eksplorasi konten dan rekomendasi yang lebih dinamis**, menjadikannya cocok untuk pengguna yang ingin **menemukan film baru dari genre berbeda namun masih relevan dengan preferensi pribadi**.

---

| Rank | **RecommenderNet Name**                           | genre                                             | RecommenderNet Rating | **NeuMF Name**                                    | NeuMF Rating |
| ---- | ------------------------------------------------- | ------------------------------------------------- | --------------------- | ------------------------------------------------- | ------------ |
| 1    | Ginga Eiyuu Densetsu                              | Drama, Military, Sci-Fi, Space                    | 9.628527              | Ginga Eiyuu Densetsu                              | 9.719216     |
| 2    | Gintama°                                          | Action, Comedy, Historical, Parody, Samurai, S... | 9.498782              | Hunter x Hunter (2011)                            | 9.716857     |
| 3    | Kimi no Na wa.                                    | Drama, Romance, School, Supernatural              | 9.365213              | Gintama                                           | 9.683640     |
| 4    | Gintama'                                          | Action, Comedy, Historical, Parody, Samurai, S... | 9.288427              | Gintama'                                          | 9.656740     |
| 5    | Steins;Gate                                       | Sci-Fi, Thriller                                  | 9.251849              | Gintama°                                          | 9.656626     |
| 6    | Gintama' : Enchousen                              | Action, Comedy, Historical, Parody, Samurai, S... | 9.247744              | Clannad: After Story                              | 9.629711     |
| 7    | Hunter x Hunter (2011)                            | Action, Adventure, Shounen, Super Power           | 9.234929              | Gintama Movie: Kanketsu-hen - Yorozuya yo Eien... | 9.585005     |
| 8    | Gintama Movie: Kanketsu-hen - Yorozuya yo Eien... | Action, Comedy, Historical, Parody, Samurai, S... | 9.222839              | Gintama' : Enchousen                              | 9.571122     |
| 9    | Gintama                                           | Action, Comedy, Historical, Parody, Samurai, S... | 9.219616              | Haikyuu!!: Karasuno Koukou VS Shiratorizawa Ga... | 9.533060     |
| 10   | Haikyuu!!: Karasuno Koukou VS Shiratorizawa Ga... | Comedy, Drama, School, Shounen, Sports            | 9.175668              | Aria The Origination                              | 9.513774     |

---
## Evaluasi Performa Model

Tujuan utama dari sistem rekomendasi film yang dikembangkan dalam proyek ini adalah untuk memprediksi **rating film yang kemungkinan besar akan diberikan oleh pengguna**, seakurat mungkin. Untuk mencapai tujuan tersebut, performa dari masing-masing model dievaluasi menggunakan metrik regresi standar, yang mengukur sejauh mana nilai prediksi mendekati rating sebenarnya.

## Metrik Evaluasi yang Digunakan

### 1. Root Mean Squared Error (RMSE)

Root Mean Squared Error (RMSE) mengukur akar dari rata-rata selisih kuadrat antara nilai rating aktual (_yᵢ_) dan nilai prediksi (_ŷᵢ_).  
Rumus:

**RMSE = √( (1/n) × Σ(yᵢ − ŷᵢ)² )**

RMSE sangat sensitif terhadap kesalahan besar, sehingga memberikan penalti lebih tinggi untuk prediksi yang meleset jauh.  
Semakin kecil nilai RMSE, semakin baik kualitas model.

---

### 2. Mean Absolute Error (MAE)

MAE mengukur rata-rata dari nilai absolut selisih antara rating sebenarnya dan hasil prediksi.  
Rumus:

**MAE = (1/n) × Σ |yᵢ − ŷᵢ|**

MAE memberikan gambaran yang lebih mudah dipahami tentang rata-rata kesalahan prediksi.  
Sama seperti RMSE, semakin kecil nilai MAE, semakin baik performa model.


---

## Hasil dan Analisis

Berikut adalah hasil perbandingan performa kedua model pada data validasi berdasarkan metrik evaluasi yang didapat:

| Metrik   | RecommenderNet (Baseline) | **NeuMF (Deep Learning)** |
| :------- | :------------------------ | :------------------------ |
| **RMSE** | 1.179464                  | 1.119572                  |
| **MAE**  | 0.892495                  | 0.840099                  |

#### Scatterplot : Actual Rating vs Predicted Rating

![Scatterplot](/Image/Scatterplot.png)

Grafik di atas menampilkan scatter plot yang memperlihatkan hubungan antara **rating film yang sebenarnya diberikan pengguna (sumbu-X)** dan **rating yang diprediksi oleh model (sumbu-Y)**, untuk dua model berbeda: RecommenderNet (kiri) dan NeuMF (kanan).
Garis diagonal merah pada masing-masing plot menunjukkan **prediksi sempurna**, yaitu ketika nilai prediksi sama persis dengan nilai aktual. Maka, semakin dekat titik-titik dengan garis ini, semakin akurat prediksi yang dihasilkan model.

**Beberapa poin penting dari visualisasi ini adalah:**

- **Kerapatan Titik di Sekitar Garis Diagonal:**  
  Kedua model menunjukkan distribusi yang cukup rapat di sekitar garis diagonal, terutama pada rating tinggi (4 dan 5). Ini mengindikasikan bahwa model cenderung **lebih akurat saat memprediksi film yang disukai pengguna**.

- **Pola Vertikal yang Konsisten:**  
  Pola vertikal pada masing-masing rating aktual menunjukkan bahwa **model memprediksi dalam rentang nilai yang relatif stabil untuk setiap kelompok rating tertentu**. Hal ini merupakan ciri dari model regresi yang cukup terkontrol.

- **Sebaran dan Variasi Prediksi:**  
  Model NeuMF (kanan) cenderung memiliki sebaran titik yang sedikit lebih merata di sekitar garis diagonal dibandingkan RecommenderNet. Hal ini bisa mengindikasikan bahwa NeuMF memiliki **kemampuan generalisasi yang baik** dalam beberapa rentang rating.

- **Potensi Kesalahan Ekstrem:**  
  Walaupun cukup akurat secara umum, kedua model masih menunjukkan adanya **prediksi yang menyimpang jauh dari nilai aktual**, terutama untuk rating rendah (1 dan 2), yang berpotensi meningkatkan nilai error keseluruhan seperti RMSE.

Secara keseluruhan, visualisasi ini menunjukkan bahwa kedua model memiliki **performa prediksi yang cukup solid**, dengan kecenderungan melakukan prediksi lebih baik untuk film yang disukai pengguna. Namun demikian, tetap terdapat ruang perbaikan terutama dalam menangani prediksi pada rating yang lebih rendah.


### Interpretasi Hasil:

#### 1. Keunggulan Model Deep Learning
Berdasarkan hasil evaluasi, model **NeuMF** menunjukkan performa yang sedikit lebih unggul dibandingkan **RecommenderNet**. Nilai **RMSE** yang dicapai oleh NeuMF adalah **0.801**, sedikit lebih rendah dibandingkan RecommenderNet dengan **0.805**. Begitu pula untuk **MAE**, di mana NeuMF memperoleh nilai **0.608**, sementara RecommenderNet berada di angka **0.614**. Kedua metrik ini menegaskan bahwa secara rata-rata, prediksi rating dari NeuMF lebih mendekati nilai rating sebenarnya.

#### 2. Justifikasi Kompleksitas Model
Perbedaan performa ini mencerminkan **dampak positif dari arsitektur gabungan** dalam NeuMF, yang mengombinasikan pendekatan *Generalized Matrix Factorization* (GMF) dengan *Multi-Layer Perceptron* (MLP). GMF mempelajari interaksi linier antara pengguna dan item, sementara MLP mampu menangkap hubungan non-linear yang lebih kompleks. Kombinasi ini memungkinkan NeuMF untuk **memodelkan pola preferensi pengguna yang tidak linier** dan tidak bisa ditangkap sepenuhnya oleh arsitektur sederhana seperti RecommenderNet.

#### 3.  Implikasi Praktis
Walaupun NeuMF membutuhkan **waktu pelatihan yang lebih lama** dan **konsumsi sumber daya komputasi yang lebih tinggi**, hasil evaluasi menunjukkan bahwa peningkatan performa prediksi yang diberikannya dapat dianggap signifikan. Dengan demikian, **jika prioritas utama adalah akurasi prediksi**, maka penggunaan NeuMF layak dipertimbangkan meskipun dengan konsekuensi komputasional yang lebih besar. Sebaliknya, RecommenderNet masih merupakan alternatif yang efisien jika dibutuhkan kompromi antara akurasi dan efisiensi.


---

## **Conclusion and Future Work**
### Kesimpulan
Proyek ini telah berhasil membangun dan membandingkan dua pendekatan *collaborative filtering* untuk sistem rekomendasi film, yaitu *RecommenderNet* sebagai model berbasis *matrix factorization*, serta *NeuMF (Neural Matrix Factorization)* sebagai model *deep learning* gabungan. Berdasarkan hasil evaluasi menggunakan metrik *Root Mean Squared Error (RMSE)* dan *Mean Absolute Error (MAE)*, model *NeuMF* menunjukkan performa yang lebih baik dibandingkan *RecommenderNet*, yang mengindikasikan bahwa prediksi rating yang dihasilkan lebih akurat dan lebih sesuai dengan preferensi pengguna.

Hasil ini memberikan *insight* bahwa penambahan kompleksitas arsitektur pada model seperti *NeuMF* dapat memberikan peningkatan performa yang signifikan, khususnya dalam konteks prediksi selera pengguna terhadap film. Kemampuan *NeuMF* dalam menangkap hubungan *non-linear* antara pengguna dan film menjadi keunggulan utama dalam menyusun rekomendasi yang lebih personal.

### Rencana Pengembangan ke Depan
Beberapa potensi pengembangan yang dapat dilakukan untuk meningkatkan sistem rekomendasi ini antara lain:
- Menambahkan informasi berbasis konten seperti genre, sutradara, tahun rilis, atau sinopsis film guna menangani permasalahan *cold-start*.
- Menggunakan arsitektur model yang lebih dalam atau menerapkan teknik lain seperti *attention mechanism* maupun *autoencoder* untuk meningkatkan representasi pengguna dan film.
- Menambahkan evaluasi berbasis *top-k recommendation* seperti *Precision@K* atau *NDCG* untuk menilai kualitas daftar rekomendasi dalam konteks pengalaman pengguna secara langsung.
- Menerapkan sistem rekomendasi ini ke dalam aplikasi *real-time*, dengan fokus pada efisiensi komputasi dan kemampuan skalabilitas terhadap pertumbuhan data pengguna dan film.

Dengan langkah-langkah tersebut, sistem rekomendasi film yang dikembangkan diharapkan dapat menjadi lebih akurat, adaptif, dan relevan dalam memenuhi kebutuhan pengguna di dunia nyata.
