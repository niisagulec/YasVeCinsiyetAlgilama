#ZaferDemirkol'un videosu yardimiyla yaptiğim calisma

import cv2
# Yüz algılama ve çizgileri çizecek fonksiyon
def highlihtFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]

    # Görüntüyü blob formatına çeviriyoruz
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0,(300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]

    # Tespit edilen her yüz için konumları hesapla ve dikdörtgen çiz
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

# Hazır modellerin eklendi

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"


# Modelin ihtiyaç duyduğu ortalama değerler
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)

# Yaş ve cinsiyet sınıfları
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Erkek','Kadin']

faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel, ageProto)
genderNet=cv2.dnn.readNet(genderModel, genderProto)

# Resmin tanıtılması algılanması ve tahmin yapılması
video=cv2.VideoCapture("kızcocuk.jpg" if "kızcocuk.jpg" else 0)
padding=20

# Sonsuz döngü içinde her kareyi işleyerek yüz, yaş ve cinsiyet tahmini yapılır
while cv2.waitKey(1)<0 :
    hasFrame, frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

# Çizgileri çizecek fonksiyonun çağırılması
    resultImg, faceBoxes= highlihtFace(faceNet, frame)
    if not faceBoxes:
        print("Yüz algılanamadı!")
    
    for faceBox in faceBoxes:
        face=frame[max(0, faceBox[1]-padding):
                   min(faceBox[3]+padding, frame.shape[0]-1),max(0, faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
        
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Cinsiyet tahmini
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Cinsiyet: {gender}')
        
        # Yaş tahmini
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Yaş: {age[1:-1]} yasinda')

        # Tahmin sonuçlarını görüntü üzerine yazdırma
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)


        # Sonuçları gösterme
        cv2.imshow("Yas ve Cinsiyet Algilama", resultImg)
