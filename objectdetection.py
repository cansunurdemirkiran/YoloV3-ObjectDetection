import cv2
thres = 0.68

dictionary = {
    "person" : "Insan",
    "bicycle" : "Bisiklet",
    "car" : "Araba",
    "motorcycle" : "Motor",
    "airplane" : "Ucak",
    "bus" : "Otobus",
    "train" : "Tren",
    "truck" : "Kamyon",
    "boat" : "Bot",
    "traffic light" : "Trafik Lambası",
    "fire hydrant" : "Yangin Muslugu",
    "street sign" : "Trafik Isareti",
    "stop sign" : "Dur Isareti",
    "parking meter" : "Parametre",
    "bench" : "Bank",
    "bird" : "Kus",
    "cat" : "Kedi",
    "dog" : "Kopek",
    "horse" : "At",
    "sheep" : "Koyun",
    "cow" : "Inek",
    "elephant" : "Fil",
    "bear" : "Ayi",
    "zebra" : "Zebra",
    "giraffe" : "Zurafa",
    "hat" : "Sapka",
    "backpack" : "Sirt Cantasi",
    "umbrella" : "Semsiye",
    "shoe" : "Ayakkabı",
    "eye glasses" : "Gozluk",
    "handbag" : "El Cantası",
    "tie" : "Kıravat",
    "suitcase" : "İs Cantası",
    "frisbee" : "Frizbi",
    "skis" : "Kayak Takımı",
    "snowboard" : "Snowboard",
    "sports ball" : "Top",
    "kite" : "Ucurta",
    "baseball bat" : "Beysbol Sopası",
    "baseball glove" : "Beysbol Eldiveni",
    "skateboard" : "Kaykay",
    "surfboard" : "Sorf Tahtası",
    "tennis racket" : "Tenis Raketi",
    "bottle" : "Sise",
    "plate" : "Tabak",
    "wine glass" : "Sarap Bardagı",
    "cup" : "Bardak",
    "fork" : "Catal",
    "knife" : "Bicak",
    "spoon" : "Kasik",
    "bowl" : "Kase",
    "banana" : "Muz",
    "apple" : "Elma",
    "sandwich" : "Sandvic",
    "orange" : "Portakal",
    "broccoli" : "Brokoli",
    "carrot" : "Havuc",
    "hot dog" : "Hot Dog",
    "pizza" : "Pizza",
    "donut" : "Donut",
    "cake" : "Kek",
    "chair" : "Sandalye",
    "couch" : "Kanepe",
    "potted plant" : "Saksı",
    "bed" : "Yatak",
    "mirror" : "Ayna",
    "dining table" : "Yemek Masası",
    "window" : "Pencere",
    "desk" : "Masa",
    "toilet" :"Tuvalet",
    "door" : "Kapi",
    "tv" : "Televizyon",
    "laptop" : "Laptop",
    "mouse" : "Fare",
    "remote" : "Tanimlanamayan",
    "keyboard": "Klavye",
    "cell phone" : "Cep Telefonu",
    "microwave" : "Mikrodalga",
    "oven" : "Fırın",
    "toaster" : "Tost Makinası",
    "sink" : "Lavobo",
    "refrigerator" : "Buzdolabı",
    "blender" : "Blender",
    "book" : "Kitap",
    "clock" : "Saat",
    "vase" : "Vazo",
    "scissors" : "Makas",
    "teddy bear" : "Oyuncak Ayı",
    "hair drier" : "Sac Kurutma Makinası",
    "toothbrush" : "Dis Fırçası",
    "hair brush" : "Tarak",
    }

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []  
classFile = 'coco.names'    
with open(classFile,'rt') as f:    
    classNames = f.read().rstrip('\n').split('\n')  

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' 
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)

net.setInputSize(320,320) 
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))	
net.setInputSwapRB(True)    

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,0,255),thickness=2)

            cv2.putText(img,dictionary[classNames[classId-1]].upper(),(box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            try:
                print("Cisim türü: " , dictionary[classNames[classId-1]])
            except:
                print("error happened" + str(classId))

    cv2.imshow("Output",img)
    cv2.waitKey(1)