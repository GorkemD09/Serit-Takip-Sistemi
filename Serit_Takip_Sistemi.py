#!/usr/bin/python
# -*- coding: utf-8 -*-

## Kaynak video (5:10-13:00): https://www.youtube.com/watch?v=jOPYpD1Knpw&t=310s
## Sonuç video: https://www.youtube.com/watch?v=kRQ0gB8dv0I

import cv2 as cv
import numpy as np

video_kaynak = r"road_test_video.mp4"

## Kameranın konumuna göre sol ve sağ şeridin x eksenine göre orta noktası.
şeritlerin_merkezi_x = 416

## Tek şeridin olması durumunda şeridi ortasını belirlemek için gerekli fark değeri.
tek_şerit_ofset = 65

## Not: Gerçek araç testinde video hızı değiştirilmez.
## Hesaplanan hıza göre video hızını değiştir. (Video: True/False, Gerçek_araç: False)
bekleme_süresi_aktif = True 
bekleme_süresi = 1 # milisaniye

## Görüntüdeki kenarları tespit eder.
def kenar_tespit(görüntü, alt_eşik_değer = 80, üst_eşik_değer = 150):
    """
    Bu fonksiyon, verilen görüntüde Canny algoritması kullanarak kenarları tespit eder.
    
    Parametreler:
    görüntü (numpy.ndarray): Kenarları tespit edilecek giriş görüntüsü.
    alt_eşik_değer (int): Canny algoritmasında kullanılan alt eşik değeri. Varsayılan değer: 80.
    üst_eşik_değer (int): Canny algoritmasında kullanılan üst eşik değeri. Varsayılan değer: 150.
    
    Dönüş:
    numpy.ndarray: Tespit edilen kenarları içeren ikili (binary) görüntü.
    """
    
    # Görüntüyü Gaussian bulanıklığı (blur) ile yumuşatarak gürültü azaltılır.
    bulanık_görüntü = cv.GaussianBlur(görüntü, (5, 5), 0)
    
    # Görüntü gri tonlamaya dönüştürülür.
    gri_görüntü = cv.cvtColor(bulanık_görüntü, cv.COLOR_BGR2GRAY)  # OpenCV BGR renk uzayını kullanır.
    
    # Canny algoritması ile kenarlar tespit edilir. 
    kenarlar = cv.Canny(gri_görüntü, alt_eşik_değer, üst_eşik_değer)
    
    return kenarlar

## Görüntüdeki istenilen bölgenin dışına maskeleme yapar.
def ilgili_bölge(görüntü):
    """
    Bu fonksiyon, verilen görüntüde koordinatlarla belirlenmiş bölgenin dışındaki
    tüm pikselleri maskeleyerek yalnızca belirtilen bölgeyi görünür kılar.
    
    Parametreler:
    görüntü (numpy.ndarray): Maskeleme işlemi yapılacak giriş görüntüsü.
    
    Dönüş:
    numpy.ndarray: Maske uygulanmış görüntü.
    """
    
    # Görüntüdeki şeritlerin olduğu bölgenin koordinatları.
    bölge = np.array([[ 
        (150, 255),  # Sol alt köşe
        (650, 255),  # Sağ alt köşe
        (550, 150),  # Sağ üst köşe
        (280, 150)   # Sol üst köşe
    ]])
    
    # Görüntünün aynı boyutunda ve siyah (0'larla) olan bir maske görüntüsü oluşturur.
    maske = np.zeros_like(görüntü)
    
    # Maskenin içine belirlediğimiz bölgeyi beyaz (255) olarak doldurur.
    # Bu sayede sadece ilgi alanındaki piksellerin üzerinde işlem yapılacak.
    cv.fillPoly(maske, bölge, (255, 255, 255))
    
    # Görüntü ve maskeyi birleştirir ve sadece istenilen bölge görünür olur.
    maskelenmiş_görüntü = cv.bitwise_and(görüntü, maske)
    
    return maskelenmiş_görüntü


## Doğrulardaki gürültüyü ayıklar.
def gürültü_ayıkla(doğrular, eşik=1):
    """
    Bu fonksiyon, verilen doğruların Z-Score değerlerini hesaplayarak belirtilen
    eşikten büyük Z-Score değerlerine sahip doğruları ayıklar.
    
    Parametreler:
    doğrular (list of tuples): Her bir doğrunun eğimi ve kesişim noktası.
    eşik (float): Z-Score eşik değeri. Varsayılan değer: 1.
    
    Dönüş:
    list of tuples: Gürültüleri ayıklanmış doğrular, eşik değerinin altındaki doğrular.
    """
    
    # Doğruların eğim ve kesişim noktaları.
    eğimler = [doğru[0] for doğru in doğrular]
    kesişimler = [doğru[1] for doğru in doğrular]
    
    # Eğimlerin ortalaması ve standart sapması hesaplanır.
    ortalama_eğim = np.mean(eğimler)
    std_eğim = np.std(eğimler)
    
    # Eğer standart sapma sıfırsa, tüm doğruların eğimi aynıdır.
    if std_eğim == 0:
        return doğrular
    
    # Z-Score hesaplanır. (eğimler için)
    z_skorları = [(eğim - ortalama_eğim) / std_eğim for eğim in eğimler]
    
    # Gürültüler ayıklanır. ( (Z-Score > eşik) olanlar gürültü olarak kabul ediliyor)
    ayıklanmış_doğrular = [(e, k) for e, k, z in zip(eğimler, kesişimler, z_skorları) if abs(z) <= eşik]
    
    return ayıklanmış_doğrular

## Doğruların eğimlerinin ve kesişim noktalarının ortalamalarını hesaplar.
def ortalama_eğim_kesişim_hesapla(doğrular):
    """
    Bu fonksiyon, verilen doğruların eğimlerinin ve kesişim noktalarının ortalamalarını hesaplar.
    
    Parametreler:
    doğrular (list of tuples): Her bir doğrunun eğimi ve kesişim noktası çiftleri.
    
    Dönüş:
    tuple: Ortalama eğim ve ortalama kesişim noktası.
    """
    
    # Doğruların eğim ve kesişim noktaları.
    eğimler = [doğru[0] for doğru in doğrular]
    kesişimler = [doğru[1] for doğru in doğrular]
    
    # Eğimlerin ve kesişim noktalarının ortalaması hesaplanır.
    ortalama_eğim = np.mean(eğimler)
    ortalama_kesişim = np.mean(kesişimler)
    
    return ortalama_eğim, ortalama_kesişim

## Doğruların ortalamasını alarak koordinatlarını belirler.
def şerit_koordinatları(doğrular):
    """
    Bu fonksiyon, verilen doğruların eğim ve kesişim noktasına göre şerit koordinatlarını hesaplar.
    
    Parametreler:
    doğrular (list of tuples): Her bir doğrunun eğimi ve kesişim noktası çiftleri.
    
    Dönüş:
    numpy array: Hesaplanan şerit çizgisinin başlangıç ve bitiş koordinatları [x1, y1, x2, y2].
    """
    
    # Hiç doğru yoksa.
    if not len(doğrular):
        return [0,0,0,0]
    
    # Gürültü ayıklama işlemi
    ayıklanmış_doğrular = gürültü_ayıkla(doğrular)
    if not len(ayıklanmış_doğrular):
        return [0,0,0,0]
    
    # Ortalama eğim ve kesişim hesaplama
    ort_eğim, ort_kesişim = ortalama_eğim_kesişim_hesapla(ayıklanmış_doğrular)
    if ort_eğim == 0:
        return [0,0,0,0]
    
    # y-koordinatları
    y1 = 255
    y2 = 150
    
    # x-koordinatları (y = m*x + c)
    x1 = int((y1 - ort_kesişim) / ort_eğim)
    x2 = int((y2 - ort_kesişim) / ort_eğim)
    
    return np.array([x1, y1, x2, y2])

## Sol ve sağ şerit çizgilerinin koordinatlarını tespit eder.
def sol_sağ_şerit_tespiti(çizgiler):
    """
    Bu foksiyon, verilen çizgilerin eğimine göre sol ve sağ olarak sınıflandırır ve
    ortalamasını alarak sol ve sağ şerit çizgilerini belirler.
    
    Parametreler:
    çizgiler (numpy.ndarray): Her bir çizginin koordinatları [x1, y1, x2, y2].
    
    Dönüş:
    tuple of numpy.ndarray: Hesaplanan şerit çizgilerinin başlangıç ve bitiş koordinatları.
        ([sol_x1, sol_y1, sol_x2, sol_y2], [sağ_x1, sağ_y1, sağ_x2, sağ_y2])
    """
    
    sol_şerit = [0,0,0,0]
    sağ_şerit = [0,0,0,0]
    
    if çizgiler is None or not len(çizgiler): # Hiç çizgi yoksa
        return np.array([sol_şerit]), np.array([sağ_şerit])
    
    sol_doğrular = []
    sağ_doğrular = []
    
    # Çizgilerin eğimine göre sınıflandırma.
    for çizgi in çizgiler:
        x1, y1, x2, y2 = çizgi.reshape(4)
        
        # Çizginin eğimi ve kesişim noktası
        eğim = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        kesişim = y1 - eğim * x1 # (y = m*x + c)
        
        if eğim < 0:  # Soldaki çizgiler
            sol_doğrular.append((eğim, kesişim))
        elif eğim > 0:  # Sağdaki çizgiler
            sağ_doğrular.append((eğim, kesişim))
    
    # Her iki şerit için koordinatlar hesaplanır.
    sol_şerit = şerit_koordinatları(sol_doğrular)
    sağ_şerit = şerit_koordinatları(sağ_doğrular)
    
    return np.array(sol_şerit), np.array(sağ_şerit)


## Sol ve sağ şeridin x eksenine göre orta noktasını hesaplar.
def şeritlerin_orta_noktasını_hesapla(sol_şerit, sağ_şerit):
    """
    Bu fonksiyon, verilen iki şerit (sol ve sağ) çizgisinin x eksenindeki son noktaların
    (görüntüye göre en üstteki nokta) arasındaki orta noktayı x ekseninde hesaplar.
    
    Parametreler:
    sol_şerit (numpy.array): Sol şeridin koordinatları (x1, y1, x2, y2).
    sağ_şerit (numpy.array): Sağ şeridin koordinatları (x1, y1, x2, y2).
    
    Döndürülen Değer:
    int: Sol ve sağ şeritlerin orta noktasının x koordinatının değeri.
    """
    
    sol_x1, sol_y1, sol_x2, sol_y2 = sol_şerit.reshape(4)
    sağ_x1, sağ_y1, sağ_x2, sağ_y2 = sağ_şerit.reshape(4)
    
    if sol_x2 == 0 and sağ_x2 == 0: # iki şerit çizgisinin olmaması
        şeritlerin_orta_noktası_x = şeritlerin_merkezi_x
    elif sol_x2 == 0: # sol şerit çizgisinin olmaması
        şeritlerin_orta_noktası_x = int(sağ_x2 - tek_şerit_ofset)
    elif sağ_x2 == 0: # sağ şerit çizgisinin olmaması
        şeritlerin_orta_noktası_x = int(sol_x2 + tek_şerit_ofset)
    else:
        şeritlerin_orta_noktası_x = int(sol_x2 + (sağ_x2 - sol_x2) / 2)
    
    return şeritlerin_orta_noktası_x

## Aracın dönmesi gereken yönü ve açıyı hesaplar.
def yön_açı_hesapla(şeritlerin_orta_noktası_x):
    """
    Bu fonksiyon, verilen iki şeridin orta noktasının x değeri ile merkez noktanın
    x değeri arasındaki farkı hesaplar ve bu farka göre aracın şerit çizgilerinin
    içinde kalabilmesi için gereken yönü ve dönüş açısını hesaplar.
    
    Parametreler:
    şeritlerin_orta_noktası_x (int): Sol ve sağ şeritlerin orta noktasının x koordinatının değeri.
    
    Döndürülen Değerler:
    tuple: 
        - metin_yön (str): Aracın dönmesi gereken yön ("Sol", "Sag", veya "Duz").
        - açı (int): Aracın dönmesi gereken açı, pozitif değer olarak.
    """
    
    piksel_derece = 4 # Her 4 piksellik farkın 1 derece olduğu varsayımı
    metin_yön = "DUZ" # Düz
    
    piksel_fark = şeritlerin_merkezi_x - şeritlerin_orta_noktası_x
    açı = int(piksel_fark / piksel_derece)
    
    if açı > 0: # Sol
        metin_yön = "SOL"
    elif açı < 0: # Sağ
        metin_yön = "SAG"
    
    return metin_yön, abs(açı)

## Aracın ve videonun hızını hesaplar.
def hız_hesapla(açı):
    """
    Bu fonksiyon, aracın dönüş açısına göre aracın hızını ve video oynatma hızını hesaplar.
    
    Parametreler:
    açı (int): Aracın dönüş açısı.
    
    Döndürülen Değer:
    int: Hesaplanan araç hızı.
    
    Açıklamalar:
    - Eğer açı küçükse, araç daha hızlı gider.
    - Eğer açı büyükse, araç daha yavaş gider. Bu sayede araç daha rahat döner.
    - Video hızını değiştirmek için `bekleme_süresi` değişkeni kullanılır ve bu, videonun oynatma hızını kontrol eder.
    
    Not:
    - Gerçek araç testinde video hızı değiştirilmez.
    """
    
    global bekleme_süresi
    
    # Açıya göre aracın hızını belirleme. (Açı arttıkca hız azalır)
    görüntü_bekle = 1 # video hızı
    hız = 5 # araç hızı
    
    # Açı aralıklarına göre video ve araç hızını belirleme
    if 3 < açı < 9:
        görüntü_bekle = 5
        hız = 4
    elif 9 <= açı < 14:
        görüntü_bekle = 9
        hız = 3
    elif 14 <= açı < 18:
        görüntü_bekle = 15
        hız = 2
    elif açı >= 18:
        görüntü_bekle = 21
        hız = 1
    
    # Video oynatma hızını değiştirme aktifse değiştir
    if bekleme_süresi_aktif:
        bekleme_süresi = görüntü_bekle
    
    return hız


## Koordinatları verilen çizgileri görüntüye çizer.
def çizgileri_çiz(görüntü, çizgiler, görüntüyü_birleştir = True):
    """
    Bu fonksiyon, verilen koordinatlarla belirtilen çizgileri görüntü veya yeni görüntü üzerine çizer.
    
    Parametreler:
    görüntü (numpy.ndarray): Çizgilerin üzerine çizileceği orijinal görüntü.
    çizgiler (numpy.ndarray): Her bir çizginin koordinatları [x1, y1, x2, y2].
    görüntüyü_birleştir (bool): Eğer True ise, çizilen çizgiler orijinal görüntü ile birleştirilir. Varsayılan değer: True.
    
    Döndürülen Değer:
    numpy.ndarray: Çizgiler çizilmiş görüntü.
    """
    
    if çizgiler is None or not len(çizgiler): # Hiç çizgi yoksa
        return görüntü
    
    # Görüntünün aynı boyutunda yeni siyah görüntü oluşturur.
    yeni_görüntü = np.zeros_like(görüntü)
    
    # Siyah görüntü üzerine çizgileri çizer.
    for çizgi in çizgiler:
        # if len(çizgi) != 4: continue
        
        x1, y1, x2, y2 = çizgi.reshape(4)
        yeni_görüntü = cv.line(yeni_görüntü, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    # Çizilen çizgiler orijinal görüntü ile birleştirilir.
    if görüntüyü_birleştir:
        yeni_görüntü = cv.addWeighted(görüntü, 0.8, yeni_görüntü, 1, 0)
    
    return yeni_görüntü

## Hız gösterge çizgilerini çizer.
def hız_göstergesi_çiz(görüntü, hız):
    """
    Bu fonksiyon, verilen hız değerine göre hız göstergesi olarak ekran üzerinde çizgiler çizer.
    Her bir çizgi, hızın bir birimini temsil eder ve hız arttıkça daha fazla çizgi eklenir.
    
    Parametreler:
    görüntü (numpy.ndarray): Çizgilerin üzerine ekleneceği orijinal görüntü.
    hız (int): Hız göstergesinin gösterdiği hız değeri. (Bu değer kadar çizgi çizilir.)
    
    Döndürülen Değer:
    None: Görüntü üzerinde değişiklik yapılır, ancak değer döndürülmez.
    """
    
    başlangıç_x = 420
    başlangıç_y = 300
    ofset_x = 12  # Her çizginin arasındaki yatay mesafe
    ofset_y = 7  # Her çizginin arasındaki boy farkı
    
    cv.putText(görüntü, "HIZ:", (350, 300), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0, 255), 2, cv.LINE_4)
    for i in range(hız):
        x1 = başlangıç_x + i * ofset_x
        y2 = başlangıç_y - (i + 1) * ofset_y
        
        cv.line(görüntü, (x1, başlangıç_y), (x1, y2), (0, 0, 255), 4)


## Görüntü işleyerek şerit takip sistemi
def Main(görüntü):
    """
    Bu fonksiyon, verilen görüntüyü işler ve şerit takip sistemi uygular.
    Şeritlerin tespiti, aracın yönü ve hızının hesaplanması gibi işlemleri içerir.
    
    Parametreler:
    görüntü (numpy.ndarray): İşlem yapılacak giriş görüntüsü.
    
    Döndürülen Değer:
    None: Bu fonksiyon, görüntü üzerinde çeşitli işlemler yapar ancak doğrudan bir değer döndürmez.
    
    Açıklamalar:
    - Görüntüdeki kenarlar tespit edilir.
    - Ardından, bu kenarlardan belirli bir bölge seçilir.
    - Hough çizgi dönüşümü ile düz çizgiler tespit edilir.
    - Sol ve sağ şeritlerin koordinatları belirlenir.
    - Şeritlerin ortası hesaplanır ve aracın yönü ile dönüş açısı hesaplanır.
    - Aracın hızı, dönüş açısına göre hesaplanır ve hız göstergesi çizilir.
    - Son olarak, tüm işlemler yapılmış görüntü ekrana gösterilir.
    """
    
    ## Görüntüdeki kenarlar tespit edilir.
    görüntü_kenarlar = kenar_tespit(görüntü)
    # cv.imshow("Kenarlar", görüntü_kenarlar)
    
    ## Kenar tespiti sonrası ilgili bölge seçilir.
    görüntü_bölge = ilgili_bölge(görüntü_kenarlar)
    # cv.imshow("İlgili Bolge", görüntü_bölge)
    
    ## Hough çizgi dönüşümü ile ilgili bölgedeki düz çizgiler tespit edilir.
    çizgiler = cv.HoughLinesP(görüntü_bölge, cv.HOUGH_PROBABILISTIC, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=5)
    # görüntü_çizgiler = çizgileri_çiz(görüntü, çizgiler)
    # cv.imshow("Cizgiler", görüntü_çizgiler)
    
    ## Sol ve sağ şeritler tespit edilir.
    sol_şerit, sağ_şerit = sol_sağ_şerit_tespiti(çizgiler)
    görüntü = çizgileri_çiz(görüntü, (sol_şerit, sağ_şerit))
    
    ## Şeritlerin orta noktasını hesaplanır ve görüntü üzerine çizilir.
    şeritlerin_orta_noktası_x = şeritlerin_orta_noktasını_hesapla(sol_şerit, sağ_şerit)
    cv.line(görüntü, (şeritlerin_merkezi_x, 153), (şeritlerin_orta_noktası_x, 153), (0, 0, 255), 2)
    
    ## Aracın yönü ve dönüş açısı hesaplanır ve görüntü üzerine eklenir.
    metin_yön, açı = yön_açı_hesapla(şeritlerin_orta_noktası_x)
    cv.putText(görüntü, str(açı) +  "'", (350, 250), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0, 255), 2, cv.LINE_4)
    cv.putText(görüntü, metin_yön, (420, 250), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0, 255), 2, cv.LINE_4)
    
    ## Aracın hızı hesaplanır.
    hız = hız_hesapla(açı)
    
    ## Aracın hızı, hız göstergesi olarak görüntü üzerine çizilir.
    hız_göstergesi_çiz(görüntü, hız)
    
    ## Sonuçlar ekranda gösterilir.
    cv.imshow("Serit Takip Sistemi", görüntü)


if __name__ == "__main__":
    
    # Video dosyasını açar.
    cap = cv.VideoCapture(video_kaynak)
    
    while True:
        # Video dosyasından her bir görüntü (frame) alınır.
        ret, görüntü = cap.read()
        
        if not ret: # video sonlandığında çık
            break
        
        # Her görüntü, şerit takip fonksiyonlarıyla işlenir.
        Main(görüntü)
        
        # Çıkmak için 'q' tuşuna bas
        if cv.waitKey(bekleme_süresi) & 0xFF == ord('q'):
            break
    
    # Video kaynağını serbest bırakır ve pencereleri kapatır.
    cap.release()
    cv.destroyAllWindows()