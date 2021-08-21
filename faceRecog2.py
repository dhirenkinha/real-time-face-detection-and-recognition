import cv2
import numpy as np
import os
from gtts import gTTS
import pyttsx3


def SpeakText(command):
      
    engine = pyttsx3.init('nsss')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty("rate", 150)
    engine.say(command) 
    engine.runAndWait()

#### knn code ###

def distance(v1,v2):
    #we will use euclidian distance
    return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
    dist = []
    
    for i in range(train.shape[0]):
        ix = train[i,:-1]
        iy = train[i,-1]
        
        #complete the distance from test point
        d = distance(test,ix)
        dist.append((d,iy))
        
    #sorting based on distance from test point
    dk = sorted(dist,key = lambda x:x[0])[:k]
    
    #retrieve only the labels
    labels = np.array(dk)[:,-1]
    
    #get the frequencies of each label
    output = np.unique(labels,return_counts = True)
    
    
    #find max frequency and corresponding label
    index = np.argmax(output[1])
    #print('output[0][index] is ',output[0][index])
    return output[0][index]

def recognise():

        #initalise camera
    cap = cv2.VideoCapture(0)

    #face detection
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


    face_data = []
    dataset_path = 'dataset/'
    labels = []

    class_id = 0 #labels for the given file
    names = {}  # mapping between id and name

    #data preparation

    for fx in os.listdir(dataset_path):
        if fx.endswith('.npy'):
            names[class_id] = fx[:-4]
            data_item = np.load(dataset_path + fx)
            face_data.append(data_item)
            
            #create labels for the class
            target = class_id*np.ones((data_item.shape[0]))
            class_id += 1
            labels.append(target)

            
    face_dataset = np.concatenate(face_data,axis = 0)
    face_labels = np.concatenate(labels,axis = 0).reshape((-1,1))

    #print(face_dataset.shape)
    #print(face_labels.shape)

    trainset = np.concatenate((face_dataset,face_labels),axis = 1)
    #print(trainset.shape)

    filePresent = 0
    pred = ""
    iteration = 0

    #testing
    while True:
        ret,frame = cap.read()
        if ret==False:
            continue
        #7 min neighbours :- for tight detection
        faces = face_cascade.detectMultiScale(frame,1.3,7)
        faces = sorted(faces,key = lambda f: f[2]*f[3])

        for face in faces[-1:]:
            x,y,w,h = face
            
            #get the face region of interest
            
            offset = 10
            face_section = frame[y-offset:y+h+offset,x - offset:x+w+offset]
            face_section = cv2.resize(face_section,(100,100))
            
            #predict label(out)
            out = knn(trainset,face_section.flatten())
            
            #display on the screen the name and rectangle around it
            #print(out)
            
            pred = names[int(out)]
            cv2.putText(frame,pred,(x,y - 10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        
        cv2.imshow('faces',frame)
        #if no face is getting detected
        if(pred==""):
            iteration += 1

            if(iteration==40):
                print("no face detected")
                break
            continue

        #voice command
        myText = " welcome " + pred + " , how are you . This is your friend robot chotu "
        SpeakText('you are ' + pred) 
        myobj = gTTS(text=myText, lang='en', slow=False)


        for fx in os.listdir("voice for face/"):
            if (fx==pred + ".mp3"):
                filePresent = 1
                print('file already exists')
                break

        if(filePresent==0):
            myobj.save("voice for face/" + pred + ".mp3")
            print('new file saved')

        print("the prediction is",pred)
        break


        key = cv2.waitKey(1)&0xFF

        
        if(key==ord('q')):
            break

    
    cap.release()
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    print("face detected")
    