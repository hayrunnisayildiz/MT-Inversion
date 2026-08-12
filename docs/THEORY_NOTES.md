## Modül 1 Teori ve Mantığı
MT yönteminde yüzeyde ölçülen Elektrik ve Manyetik alanlar arasındaki ilişki, frekans ortamında karmaşık empedans tenzörü(Z) ile ifade edilir.

Bir dosyadan okuduğumuz ham veriyi kodumuzda tek tek değişkenler halinde gezdirmek yerine, Julia'nın güçlü struct özelliğini kullanarak MTData adı altında paketleyeceğiz.

Bu modül şunları içerecek:
-İstasyon Meta Verileri: Enlem, boylam, yükseklik, istasyon adı.
-Frekans Vektörü
-Empedans Tenzörü: Her frekans için karmaşık sayılardan oluşan 2x2 matrisler dizisi.

## DataIngestion.jl
Yapay zeka veya ters çözüm algoritmaları düz metin dosyalarından anlamaz, karmaşık matrisler ve matematiksel dizilerle çalışır.
Bu modülün temel amacı: dosyadaki ham yazıları alıp, fiziğin ve yapay zekanın işleyebileceği matematiksel nesnelere dönüştürmektir.

Ham dosya
-
1. Koordinatları&Yüksekliği çeker.
2. Frekansları sıralar.
3. Empedans sayılarını karmaşık sayılara dönüştürür.
-
MTData paketleme nesnesi (struct)
-Görünür özdirenç ve faz hesaplanır.

2. MT fiziğinde direnç ve faz bilgisi karmaşık sayılarda gizlidir. Bu yüzden Zxy için yazan iki adet reel sayıyı alıp Julia'nın anlayacağı Zxy=Real + i x Imag biçimine getirir.

3. Arazide cihaz doğrudan yerin özdirencini ölçemez. Voltaj ve manyetik alan ölçer. Bu yüzden empedans alınır fizik formülüyle yeraltının görünen özdirenç ve faz değerine dönüştürür.

Özetle: DataIngestion modülü, data/raw/ altındaki .edi/.txt dosyasından istasyon lokasyonunu, frekans dizisini ve karmaşık empedans tenzörünü ($Z$) okur. Bu verileri MTData yapısında nesneleştirir ve inversiyon kayıp (loss) fonksiyonunda kullanılmak üzere görünür özdirenç ile faz değerlerini otomatik türetir.


Görünür Özdirenç: Yer kabuğunun o frekansın nüfuz ettiği derinlikteki ortalama direnç cevabıdır. 

Faz: Elektrik ve manyetik alanlar arasındaki açı farkıdır. Homojen yer altında faz 45 derecedir.
fazın 45 dereceden büyük olması derinlere indikçe ortamın daha iletken bir ortama geçtiğini gösterir.


## Modül 2
Inversion algoritmalarında yer altını doğrudan sürekli bir ortam olarak modelleyemeyiz. hacmi küçük hücrelere böleriz. Prior Model ise ters çözüme başlamadan önce yer altına atadığımız başlangıç özdirenç dağılımıdır.

Bu modülün görevleri:
3B Izgara tanımlama
Skin Depth Hesabı: Elektromanyetik dalgalar derine indikçe zayıfladığı için Z eksenindeki hücre boyuları derne doğru katlanarak büyütülür.
Özdirenç Matrsi ve Kısıtlar: Tüm hücrelere bölgesel arka plan özdirenci atanır ve fiziki sınır kısıtları belirlenir.

PriorModels modülü yapay zekanın veya optimizasyon algoritmasının yer altında arama yapacağı 3 boyutlu fiziksel mesh alanını ve başlangı. hipotezini inşa eder.

Yer altı sonsuz ve kesintisiz bir ortamdır. Bilgisayar bunu hesaplasın diye küçük hücrelere bölmek zorundayız.

Yüzeyde ince derinde kalın hücreler kullanıyoruz. MT yönteminde kullanılan elektromanyetik dalgalar yüzeyde yüksek çözünürlüğe sahiptir, ancak derine indikçe enerjilerini kaybederler. Yüzeyde çözünürlük yüksektir bu yüzden ilk katman kalınlığı küçük seçilir. Derine indikçe çözünürlük düşer, dalga derindeki 10 metrelik değişimi fark edemez. Bu yüzden gereksiz hesaplama yükünden kaçınmak ve fiziğe uymak için her katman bir öncekinden daha kalın yapılır. 


Modül 3:

Jeofizikte ileri modelleme, elindeki yeraltı özdirenç modelinden hareket ederek yüzeyde ölçülecek teorik empedans değerini hesaplayan fizik motorudur. 

Inversion kalbinde bu motor yatar.
1.Bir özdirenç modeli verilir.
2.Maxwell denklemlerine dayalı MT ileri modelleme çözülür.
3.Hesaplanan Zpred ile .edi dosyasından okuduğumuz gerçek Zobs karşılaştırılır.

özdirenç modeli-forward model- teorik empedans-loss hesabı-gerçek veri Z obs


Inversion tam olarak burada devreye girecek. 
Sinir ağı veya optimizasyon algoritması bu başlangıç modelini alıp hücre hücre güncelleyecek ve ürettiği teorik yanıtı gerçek TTT07 verisine yaklaştırmaya çalışacak. 

Araziden aldığımız yüzey verisinden hareket ederek yeraltının 3d haritasını geriye doğru hesaplamaya inversion denir. 

Inversion matematikte dünyanın en zor problemlerinden biridir çünkü Çoklu Çözüm problemi vardır. Yüzeydeki aynı Z verisini üretebilecek trilyonlarca farklı yeraltı katman kombinasyonu olabilir.

## YZ işte tam bu noktada işe yarar:
# Arama uzayını daraltır: YZ jeolojik olarak imkansız olan mantıksız derecede aşırı yüksek veya eksi dirençli çözümleri eler. Jeolojik ön modelimize en yakın 3B yeraltı modelini bulur.

# 6000 tane hücreyi aynı anda optimizasyonla günceller. Lux.jl ve Zygote.jl her adımda 6000 hücrenin değerini milisaniyeler içinde ayarlayarak yüzeydeki gerçek ttt07 verisine uydurur. 

# Geleneksel matris tabanlı ters çözüm yöntemleri saatlerce/günlerce sürebilir. Fizik tabanlı yapay zeka türevleri otomatik alarak yer altı modelini çok daha hızlı ve kararlı bir şekilde kestirir.

Bilinen: Arazide ölçülen yüzey verisi

Başlangıç: 100 Ω·m prior model -forward model - z_pred tahmini

SciML Inversion:
Z_pred ile Z_obs arasındaki farka bakar. 
Hücrelerin özdirençlerini adım adım değiştirir.
Fark sıfıra yaklaşana kadar modeli günceller.

Çıktı: Yeraltının 3B gerçek özdirenç haritası


Modül 4: SciMLInversion.jl
Yeraltındaki özdirenç modelimizi yapay zekanın optimize edebileceği parametreler olarak tanımlarız.
Data Misfit: Forward modelden çıkan Z pred ile arazideki Zobs arasındaki fark.
Regularization: Yeraltı özdirençlerinin jeolojik olarak mantıksız değerlere kaçasını önleyen, ön modele bağlı tutan kısıt terimi.
Otomatik türev (Zygote) kullanarak her adımda Loss'u düşürecek şekilde yeraltı özdirençlerini günceller. 


İLK DENEME
julia --project=. src/SciMLInversion.jl ile TTT07 üzerinde 20 adım koşuyor; Loss ≈ 25.94 → ≈ 25.94 (çok küçük düşüş). Pipeline doğru; optimizasyon henüz veriyi anlamlı şekilde yakalamıyor.

Neden sınırlı?
Forward 1D merkez kolon → ~15 hücre data’ya bağlanıyor, 3B harita henüz yok.
|Z_obs| ≫ |Z_pred| (100 Ω·m prior) → büyük, neredeyse sabit misfit.
Gradyan çok küçük + sabit lr → adım başına ρ değişimi ~%0.1.

