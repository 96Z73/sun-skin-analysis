import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageSequence
from tkinter import filedialog
from functools import partial
import time
import datetime
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.cluster import MiniBatchKMeans as KMeans
from collections import Counter
import pprint
from matplotlib import pyplot as plt
import imutils

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

class CustomError(Exception):
    pass

class SkinStuff:
    def __init__(self) -> None:
        self.hasBlack = False
        self.lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
        self.upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
        self.toggle_frame_flag = True
        self.toggle_frame_flag2 = True
        
        self.face_detection = mp_face_detection.FaceDetection()
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.createScreen()

    def detect_face(self, frame):
        # Convert the image to RGB format
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces in the image
        results = self.face_detection.process(image_rgb)
        # Draw bounding boxes around the detected faces
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                print("BBOX CONTENTS ARE ", bbox)
                ih, iw, _ = frame.shape
                xmin = int(bbox.xmin * iw)
                ymin = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                cv2.rectangle(frame, (xmin, ymin), (ymin + w, ymin + h), (0, 255, 0), 2)

                # Draw facial landmarks
                landmarks = self.face_mesh.process(image_rgb)
                if landmarks.multi_face_landmarks:
                    for face_landmarks in landmarks.multi_face_landmarks:
                        for i, landmark in enumerate(face_landmarks.landmark):
                            x = int(landmark.x * iw)
                            y = int(landmark.y * ih)
                            # cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

                            # Check for dark circles around the eyes
                            if i in [33, 133]:  # Indices for the right and left eye landmarks
                                eye_radius = int(h * 0.08)  # Adjust the radius based on the size of the face

                                # Extract the ROI around the eye
                                eye_roi = frame[max(0, y - eye_radius):min(y + eye_radius, ih),
                                                max(0, x - eye_radius):min(x + eye_radius, iw)]
                                if eye_roi.size != 0:  # Check if the eye ROI exists
                                    # Convert the ROI to grayscale for wrinkle detection
                                    eye_roi_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

                                    # Perform wrinkle detection on the eye ROI
                                    edges = cv2.Canny(eye_roi_gray, 100, 150)
                                    number_of_edges = np.count_nonzero(edges)
                                    if number_of_edges > 70:
                                        cv2.putText(frame, "Wrinkle Found", (xmin, ymin - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    else:
                                        cv2.putText(frame, "No Wrinkle Found", (xmin, ymin - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                                    # Check for dark circles
                                    mean_intensity = cv2.mean(eye_roi_gray)[0]
                                    if mean_intensity < 80:  # threshold based on lighting conditions
                                        cv2.putText(frame, "Dark Circle", (xmin, ymin - 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    else:
                                        cv2.putText(frame, "No Dark Circle", (xmin, ymin - 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    
    def capturei(self):
        # Create the label to display the camera feed
        self.Video_label = tk.Label(self.frame)
        self.Video_label.pack()
        self.Video_label.place(x=60, y=10)

        def scc_image():
            # Disable the screenshot button
            self.screenshot_btn.config(state=tk.DISABLED)

            # Capture the current frame from the camera
            _, frame = camera.read()
            # Generate a unique image file name
            image_path = f"captured_{datetime.datetime.now().strftime('image')}.png"
            # Save the image
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")

            # Process the captured image
            self.capturedimage(image_path)

            # Release the camera
            camera.release()

            # Remove the video label
            self.Video_label.pack_forget()
            self.Video_label.destroy()
            # Remove the screenshot button
            self.screenshot_btn.pack_forget()
            self.screenshot_btn.destroy()

        self.screenshot_btn = tk.Button(self.frame, text="screenshot", command=scc_image)
        self.screenshot_btn.pack(pady=55)

        camera = cv2.VideoCapture(0)
        while True:
            _, frame = camera.read()
            if not _:
                return
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.Video_label.imgtk = imgtk
            self.Video_label.configure(image=imgtk)
            self.frame.update()


   
    # Function to open file dialog and select image file
    def uploadi(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            frame = cv2.imread(file_path)
            frame = imutils.resize(frame,500,500)
            self.display_frame(frame)

    def capturedimage(self, image_path):
        if image_path:
            frame = cv2.imread(image_path)
            self.display_frame(frame)

    def display_frame(self, frame):

        self.Video_label2 = tk.Label(self.frame)
        self.Video_label2.pack()
        self.Video_label2.place(x=60, y=10) 

        def endd():
            self.Video_label2.pack_forget()
            self.Video_label2.destroy()
            self.gb.pack_forget()
            self.gb.destroy()
            self.Video_label2.pack_forget()
            self.Video_label2.destroy()
            self.frame3.pack(fill='both', expand=True)
        
        output_frame = self.detect_face(frame)
        img = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.Video_label2.imgtk = imgtk
        self.Video_label2.configure(image=imgtk)

        self.gb = tk.Button(self.frame, text="go back", command=endd)
        self.gb.pack(pady=10)

        

        
   

    def extractSkin(self, image):
        # Taking a copy of the image
        img = image.copy()
        # Converting from BGR Colours Space to HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Defining HSV Thresholds
        
        # Single Channel mask, denoting presence of colors in the specified threshold
        skinMask = cv2.inRange(img, self.lower_threshold, self.upper_threshold)
        # Cleaning up mask using Gaussian Filter
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        # Extracting skin from the threshold mask
        skin = cv2.bitwise_and(img, img, mask=skinMask)
        # Return the Skin image
        return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
    

    def removeBlack(self, estimator_labels, estimator_cluster):
        # Get the total number of occurrences for each color
        occurrence_counter = Counter(estimator_labels)
        # Loop through the most common occurring colors
        for x in occurrence_counter.most_common(len(estimator_cluster)):
            # Quick list comprehension to convert each RGB number to int
            color = [int(i) for i in estimator_cluster[x[0]].tolist()]
            # Check if the color is [0,0,0], indicating black
            if color == [0, 0, 0]:
                del occurrence_counter[x[0]]
                hasBlack = True
                estimator_cluster = np.delete(estimator_cluster, x[0], 0)
                break
        
        return (occurrence_counter, estimator_cluster, hasBlack)


    def getColorInformation(self, estimator_labels, estimator_cluster, hasThresholding=False):
        # Variable to keep count of the occurrence of each color predicted
        occurrence_counter = None
        # Output list variable to return
        colorInformation = []
        # If a mask has been applied, remove the black color
        if hasThresholding == True:
            (occurrence, cluster, black) = self.removeBlack(
                estimator_labels, estimator_cluster)
            occurrence_counter = occurrence
            estimator_cluster = cluster
            # Check for Black
            hasBlack = black
        else:
            occurrence_counter = Counter(estimator_labels)
        # Get the total sum of all the predicted occurrences
        totalOccurrence = sum(occurrence_counter.values())
        # Loop through all the predicted colors
        for x in occurrence_counter.most_common(len(estimator_cluster)):
            index = (int(x[0]))
            # Quick fix for index out of bounds when there is no threshold
            index = (index-1) if ((hasThresholding & hasBlack) \
                                & (int(index) != 0)) else index
            # Get the color number into a list
            color = estimator_cluster[index].tolist()
            # Get the percentage of each color
            color_percentage = (x[1]/totalOccurrence)
            # Make the dictionary of the information
            colorInfo = {"cluster_index": index, "color": color,
                        "color_percentage": color_percentage}
            # Add the dictionary to the list
            colorInformation.append(colorInfo)

        return colorInformation

    def extractDominantColor(self, image, number_of_colors=3, hasThresholding=False):
        # Quick Fix: Increase cluster counter to neglect the black (Read Article)
        if hasThresholding == True:
            number_of_colors += 1
        # Take a copy of the image
        img = image.copy()
        # Convert the image into RGB color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Reshape the image
        img = img.reshape((img.shape[0]*img.shape[1]), 3)
        # Initialize the KMeans object
        timeA = time.time()
        estimator = KMeans(n_clusters=number_of_colors, random_state=0)
        # Fit the image
        estimator.fit(img)
        # Get color information
        print(f"INSIDE extractDominantColor TIME for FIT WAS: {time.time() - timeA}")
        timeB = time.time()
        colorInformation = self.getColorInformation(
            estimator.labels_, estimator.cluster_centers_, hasThresholding)
        print(f"INSIDE extractDominantColor TIME for getColorInformation WAS: {time.time() - timeB}")
        return colorInformation


    # Create a 500x100 black image
    def plotColorBar(self, colorInformation):
        color_bar = np.zeros((100, 500, 3), dtype="uint8")
        # Get the most dominant color
        dominant_color = colorInformation[0]["color"]
        dominant_color = tuple(map(int, dominant_color))
        # Fill the color bar with the dominant color
        cv2.rectangle(color_bar, (0, 0), (color_bar.shape[1], color_bar.shape[0]), dominant_color, -1)
        return color_bar


    def calculateTypologyAngle(self, color_info):
        ita_values = []
        for color in color_info:
            # r, g, b = color['color']
            lab = cv2.cvtColor(np.uint8([[color['color']]]), cv2.COLOR_RGB2LAB)
            l_prime = lab[0][0][0] / 02.55
            a_prime = (lab[0][0][1] - 128) 
            b_prime = (lab[0][0][2] - 128) 
            ita = np.arctan2((l_prime- 50),b_prime) * (180 / np.pi)
            ita_values.append(ita)

        return ita_values

    def pretty_print_data(self, color_info):
        for x in color_info:
            print(pprint.pformat(x))
            print()


    def capture_image(self):
        # Create the label to display the camera feed
        self.Video_label = tk.Label(self.frame)
        self.Video_label.pack()
        self.Video_label.place(x=60, y=10)

        def scc_image():
            # Disable the screenshot button
            self.screenshot_btn.config(state=tk.DISABLED)

            # Capture the current frame from the camera
            _, frame = camera.read()
            # Crop the image inside the rectangle
            image = frame[y:y+h, x:x+w]
            # Generate a unique image file name
            image_path = f"captured_{datetime.datetime.now().strftime('image')}.png"
            # Save the image
            cv2.imwrite(image_path, image)
            print(f"Image saved as {image_path}")

            # Process the captured image
            self.process_image(image)

            # Release the camera
            camera.release()

            # Remove the video label
            self.Video_label.pack_forget()
            self.Video_label.destroy()

            # Remove the screenshot button
            self.screenshot_btn.pack_forget()
            self.screenshot_btn.destroy()

        self.screenshot_btn = tk.Button(self.frame, text="screenshot", command=scc_image)
        self.screenshot_btn.pack(pady=55)

        camera = cv2.VideoCapture(0)
        while True:
            _, frame = camera.read()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            x, y, w, h = 220, 140, 200, 200  # coordinates and size of the rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draws a rectangle in the frame
            hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

            bgr = cv2.cvtColor(frame, cv2.COLOR_HLS2BGR)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.Video_label.config(image=imgtk)
            self.frame.update()

   
    # Function to open file dialog and select image file
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            self.process_image(image)
  


    # Process the captured/uploaded image
    def process_image(self, image):
        skin = self.extractSkin(image)
        dominantColors = self.extractDominantColor(skin, hasThresholding=True)
        print("Color Information")
        self.pretty_print_data(dominantColors)

        ita_values = self.calculateTypologyAngle(dominantColors)

        # Determine the type based on ITA
        types = []
        for i, ita in enumerate(ita_values):
          if i+1 == 1:
            if ita > 55:
                types.append("I")
            elif 41 < ita < 55:
                types.append("II")
            elif 28 < ita < 41:
                types.append("III")
            elif 10 < ita < 28:
                types.append("IV")
            elif -30 < ita < 10:
                types.append("V")
            elif ita < -30:
                types.append("VI")
            
        spf = " "
        simg = None

        for i, ita in enumerate(ita_values):
          if i+1 == 1:
            if ita > 55:
                spf = "30+ SPF"
                simg = Image.open("media/type1.png")

            elif 41 < ita < 55:
                spf = "30+ SPF"
                simg = Image.open("media/type2.png")

            elif 28 < ita < 41:
                spf = "30+ SPF"
                simg = Image.open("media/type3.png")

            elif 10 < ita < 28:
                spf = "30+ SPF"
                simg = Image.open("media/type4.png")

            elif -30 < ita < 10:
                spf = "15+ SPF"
                simg = Image.open("media/type5.png")

            elif ita < -30:
                spf = "15+ SPF"
                simg = Image.open("media/type6.png")

        # Update the UI with the dominant color information and color bar
        self.update_ui(dominantColors, types, spf, simg)


    # Update the UI with the dominant color information and color bar and spf
    def update_ui(self, dominantColors, types, spf, simg):

        self.info_text.delete("1.0", tk.END)
        self.canvas.delete("all")
       
        # Display the types
        self.info_text.insert(tk.END, f"Analysis outcome:\n")
        for i, color_type in enumerate(types):
            self.info_text.insert(tk.END, f"\nYour skin type is: {color_type}\n")
        self.info_text.insert(tk.END, f"\nYour skin recommends: {spf}\n")
        self.info_text.insert(tk.END, f"\n\nBest Clothing Color:\n")
        self.info_text.insert(tk.END, "\n")
        

        def ccolor():

            if simg is not None:
                # Resize the image
                resized_img = simg.resize((490, 100))  
                img = ImageTk.PhotoImage(resized_img)
                self.img_label.configure(image=img)
                self.img_label.image = img

        ccolor()

        # Create the color bar
        color_bar = self.plotColorBar([dominantColors[0]])
        color_bar_image = Image.fromarray(color_bar)
        color_bar_image = ImageTk.PhotoImage(color_bar_image)

        # Update the canvas with the color bar image
        self.canvas.config(width=color_bar.shape[1], height=color_bar.shape[0])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=color_bar_image)
        self.canvas.image = color_bar_image
    

    def toggle_frame(self):
        if self.toggle_frame_flag:
            self.frame2.pack(fill='both', expand=True)
            self.gif_label.pack_forget()
            self.text_label.pack_forget()
            self.text_label2.pack_forget()

        else:
            self.frame3.pack(fill='both', expand=True)
            self.canvas.pack_forget()
            self.capture_button2.pack(pady=10)
            self.upload_button2.pack(pady=10)
            self.button3.pack()
        self.toggle_frame_flag = not self.toggle_frame_flag

    def toggle_frame2(self):
        if self.toggle_frame_flag2:
            self.frame3.pack_forget()
            self.frame2.pack(fill='both', expand=True)
            self.canvas.pack()
        self.toggle_frame_flag2 = not self.toggle_frame_flag2
    
        
    def createScreen(self):
        # Create the Tkinter window

        self.frame = tk.Tk()
        self.frame.maxsize(800, 600)
        self.frame.minsize(800, 600)
        button1 = tk.Button(self.frame, text="start", command=self.toggle_frame)
        button1.pack()
        button1.place(x=375, y=450)
        self.frame.title("Color Analysis")
        self.frame.geometry("800x600")
        self.frame.configure(bg="#88cffa")

        # Load the animated GIF image with transparency
        gif_image = Image.open("media/sun.gif")
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif_image)]

        # Display the animated GIF
        self.gif_label = tk.Label(self.frame, bg="#88cffa")
        self.gif_label.pack()

        self.current_frame = 0
        self.photo_image = None

        def update_frame():
            frame = frames[self.current_frame]
            frame = frame.resize((200, 200)) 
            self.photo_image = ImageTk.PhotoImage(frame)
            self.gif_label.configure(image=self.photo_image)
            self.current_frame = (self.current_frame + 1) % len(frames)
            self.frame.after(100, update_frame)

        update_frame()
        
        # Create a label for the text
        self.text_label = tk.Label(self.frame, text="SunSkin", font=("Arial", 14), bg="#88cffa")
        self.text_label.pack(pady=10)

        self.text_label2 = tk.Label(self.frame, text="A simple skin tone analysis app based on the Fitzpatrick scale", font=("Arial", 14), bg="#88cffa")
        self.text_label2.pack(pady=10)
        self.text_label2.place(relx = 0.5, rely = 0.5, anchor = 'center')
     
        self.frame2 = tk.Frame(self.frame)
        self.frame2.configure(bg="#88cffa")
    

        # Create the capture button
        self.capture_button = tk.Button(self.frame2, text="Capture Image", command=self.capture_image)
        self.capture_button.pack(pady=10)
        self.capture_button.place(x=350, y=400)

        # Create the upload button
        self.upload_button = tk.Button(self.frame2, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)
        self.upload_button.place(x=352, y=450)

        # Create the text area to display color information
        self.info_text = tk.Text(self.frame2, width=60, height=15)
        self.info_text.pack(pady=10)

        # Create the canvas to display the color bar
        self.canvas = tk.Canvas(self.frame2, bg="white", width=500, height=105)
        self.canvas.pack()

        # Create a label widget to display the image within the frame
        self.info_label = tk.Label(self.info_text, bg="white")
        self.info_label.pack(pady=150)
        self.info_label.place(relx = 0.5, rely = 0.5, anchor = 'center')
        self.info_label.place(x= 1, y= 70)
        self.img_label = tk.Label(self.info_label, bg="white")
        self.img_label.place(x= 1, y= 70)
        self.img_label.pack()

        
        #frame3
        self.button2 = tk.Button(self.frame2, text="get face analysis", command=self.toggle_frame)
        self.button2.pack()
        self.button2.place(x=345, y=500)

        self.frame3 = tk.Frame(self.frame)
        self.frame3.configure(bg="#88cffa")


        self.button3 = tk.Button(self.frame3, text="go back", command=self.toggle_frame2)
        self.button3.pack()
        self.button3.place(x=350, y=400)

        self.upload_button2 = tk.Button(self.frame3, text="Upload Image", command=self.uploadi)
        self.upload_button2.pack(pady=10)

        self.capture_button2 = tk.Button(self.frame3, text="Capture Image", command=self.capturei)
        self.capture_button2.pack(pady=5)




if __name__ == '__main__':

    obj = SkinStuff()
    # Run the Tkinter event loop
    obj.frame.mainloop()