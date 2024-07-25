import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import pyttsx3
import threading

def perform_object_detection_and_tts(image_path):
    img = cv2.imread(image_path)

    classNames = []
    classFile = "coco.names"
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)

    if len(classIds) != 0:
        text = pyttsx3.init()
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box,	(139,0,0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255.0), 2)
            cv2.imshow("output", img)
            cv2.waitKey(1)
            ans = classNames[classId - 1]
            rate = 40
            text.setProperty('rate', rate)
            text.say(ans)
            text.runAndWait()

    
    cv2.destroyAllWindows()
 

def perform_live_object_detection_and_tts():
    obj_name = pyttsx3.init()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open Camera.")
        return

    camera.set(3, 840)
    camera.set(4, 680)

    classnames = []
    classfile = 'coco.names'

    with open(classfile, 'rt') as f:
        classnames = f.read().rstrip('\n').split('\n')

    configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightpath, configpath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        ret, img = camera.read()
        if not ret:
            print("Error: could not read frame.")
            break
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        print(classIds, bbox)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(128, 128, 128), thickness=2)
                cv2.putText(img, classnames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255.0), 2)

        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(classIds) != 0:
            ans = classnames[classId - 1]
            rate = 40
            obj_name.setProperty('rate', rate)
            obj_name.say(ans)
            obj_name.runAndWait()

    camera.release()
    cv2.destroyAllWindows()

def upload_image():
    file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.gif")])
    if file_path:
        threading.Thread(target=perform_object_detection_and_tts, args=(file_path,)).start()

def live_detection():
    threading.Thread(target=perform_live_object_detection_and_tts).start()

def main():
    root = tk.Tk()
    root.title("Object Detection System")
    
    root.configure(background="pink")  
    
    upload_button = tk.Button(root, text="Upload Image", bg="white", fg="brown", command=upload_image)
    upload_button.pack(pady=30, anchor='center')  

    # Add a live detection button
    live_detection_button = tk.Button(root, text="Live Detection", bg="white", fg="brown", command=live_detection)
    live_detection_button.pack(pady=30, anchor='center')

    # Add an exit button
    exit_button = tk.Button(root, text="Exit", bg="white", fg="brown", command=root.destroy)
    exit_button.pack(pady=30, anchor='center') 

    image_path = "C:\\Users\\nf752\\OneDrive\\Desktop\\ZOYA\\OTTS\\BLIND-PERSON ASSISTANCE\\background.jpeg"
    pil_image = Image.open(image_path)
    background_image = ImageTk.PhotoImage(pil_image)
    
    # # Create a label widget to display the image
    label = tk.Label(root, image=background_image)
    label.pack()
    root.mainloop()

if __name__ == "__main__":
    main()
