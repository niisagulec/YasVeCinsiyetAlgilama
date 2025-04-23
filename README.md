# Yaş Ve Cinsiyet Algılama

## Amacımız
Resimdeki bir yüzün yaşını ve cinsiyetini algılayıp tahmin etmektir.

## Kullanılan Teknolojiler 
1. **Python**: Proje, Python programlama dili ile geliştirilmiştir.
   
2. **OpenCV**: Görüntü işleme ve bilgisayarla görme için kullanılan popüler bir kütüphane. OpenCV, yüz tespiti, resim işleme ve çizim işlemleri gibi görevlerde kullanılmıştır.

3. **DNN (Deep Neural Network)**: OpenCV'nin derin öğrenme modülü, yüz tespiti, yaş ve cinsiyet tahmini gibi görevlerde kullanılan önceden eğitilmiş derin öğrenme modellerini yüklemek ve kullanmak için kullanılmıştır.

4. **Caffe ve Prototip Model Dosyaları**:
   - `opencv_face_detector.pbtxt` ve `opencv_face_detector_uint8.pb`: Yüz tespiti için kullanılan model dosyaları.
   - `age_deploy.prototxt` ve `age_net.caffemodel`: Yaş tahmini için kullanılan Caffe tabanlı model dosyaları.
   - `gender_deploy.prototxt` ve `gender_net.caffemodel`: Cinsiyet tahmini için kullanılan Caffe tabanlı model dosyaları.

5. **NumPy**: Sayısal hesaplamalar ve matris işlemleri için kullanılan kütüphane. NumPy, görüntü verisi üzerinde blob oluşturma ve işlem yapma işlemlerinde kullanılmıştır.

