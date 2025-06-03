import os
import pickle
import shutil
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import imageio
import threading
import time
from feat_extraction import _features

def canny_edge_detection(image, low_threshold=100, high_threshold=200):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


def prepare_features(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Apply Canny Edge Detection
    edges = cv2.Canny(image, threshold1=100, threshold2=200)

    edges = edges.astype(np.float32)

    return edges


# ------------------------------
# Prediction Function
# ------------------------------
def predict_(model,input_std):
    return model[input_std]
def predict_with_trained_model(model, feat):
    predictions = []
    cnt = 0
    delete_folder_if_exists('Prediction')
    if not os.path.exists('Prediction'):
        os.makedirs('Prediction')
    for feat_ in feat:
        cnt+=1
        input_feat =  feat_
        # print(input_std)
        # print(type(model))
        # print(model.keys)
        if feat_ in model:
            predicted_mask = predict_(model,input_feat)
            print(cnt)
        else:
            closest_std = min(model.keys(), key=lambda k: abs(k - feat_))
            predicted_mask = predict_(model,closest_std)
        # predicted_mask = cv2.resize(predicted_mask,(150,150))
        cv2.imwrite('Prediction//predicted_mask'+str(cnt)+'.png',predicted_mask)
        predictions.append(predicted_mask)
    return predictions

# Check if the folder exists and delete it if so
def delete_folder_if_exists(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # This deletes the folder and all its contents


class GIFPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GIF Player")  # Set window title
        self.root.geometry("800x400")  # Set window size (width x height)
        self.root.configure(bg='#90EE90')  # Set background color of the window
        self.root.resizable(False, False)  # Disable window resizing
        self.root.title("GIF Player")

        # Frames for GIFs
        self.frame1 = tk.Label(self.root, bg="#90EE90", fg="white")
        self.frame1.place(x=70, y=70 )

        self.frame2 = tk.Label(self.root,  bg="#90EE90", fg="white")
        self.frame2.place(x=340, y=70 )

        # Buttons
        self.load_button1 = tk.Button(self.root, text="Load Stereogram", command=self.load_gif1)
        self.load_button1.place(x=40, y=310 )

        self.load_button2 = tk.Button(self.root, text="Predict Hidden pattern", command=self.load_gif2)
        self.load_button2.place(x=220, y=310 )

        # self.play_button = tk.Button(self.root, text="Play Both GIFs", command=self.play_gifs)
        # self.play_button.place(x=420, y=310 )

        # GIFs placeholders
        self.gif1_path = None
        self.gif2_path = None

    def load_gif1(self):
        self.gif1_path = filedialog.askopenfilename(filetypes=[("GIF files", "*.gif")])
        if self.gif1_path:
            input_steregram = self.gif1_path

            def _Process(input_steregram):
                def convert_gifs_in_folder(input_steregram, output_folder):
                    cnt = 0
                    gif = imageio.mimread(input_steregram)
                    # Save each frame as an individual image
                    for i, frame in enumerate(gif):
                        cnt += 1
                        print(cnt)
                        # print(frame.shape)
                        frame_path = output_folder + '//image' + str(cnt) + '.png'
                        imageio.imwrite(frame_path, frame)

                output_folder = 'frames'
                if not os.path.exists('frames'):
                    os.makedirs('frames')
                # Convert all GIFs in the 'Stereograms' folder
                convert_gifs_in_folder(input_steregram, output_folder)

            # Path to the folder
            stereogram_folder = 'frames'

            # Delete the folder if it exists
            delete_folder_if_exists(stereogram_folder)

            # Now create the folder again
            os.makedirs(stereogram_folder)

            _Process(input_steregram)
            feat = _features(stereogram_folder)

            def save_gif1(images, gif_path, duration=500, loop=1):
                """
                Save a list of images as a GIF with looping set to 'loop'.
                :param images: List of image arrays or PIL images.
                :param gif_path: File path for saving the GIF.
                :param duration: Duration for each frame in milliseconds.
                :param loop: Number of times the GIF loops. 0 = infinite.
                """
                # Ensure images are numpy arrays and normalized for display
                image_list = []
                for img in images:
                    # if isinstance(img, np.ndarray):
                    #     img = (img * 255).astype(np.uint8)  # Normalize to 0-255
                    img = cv2.resize(cv2.imread('frames//'+img),(150,150))
                    img = Image.fromarray(img)
                    image_list.append(np.array(img))
                imageio.mimsave(gif_path, image_list, duration=500, loop=0)
                print(f"GIF saved to {gif_path}")
                del images

            self.gif1_path = 'input.gif'
            save_gif1(os.listdir(stereogram_folder), 'input.gif', duration=500, loop=1)
            np.save('pre_evaluation/feat',feat)
            self.frame1_ = tk.Label(self.root, text="Input", bg="black", fg="white")
            self.frame1_.place(x=130, y=20)
            self.display_gif_frame(self.gif1_path, self.frame1)
            self.play_gifs()

    def load_gif2(self):
        feat = np.load('pre_evaluation//feat.npy')
        # Load the saved model
        with open('pre_evaluation//model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        images = predict_with_trained_model(loaded_model, feat)

        def save_gif(images, gif_path, duration=500, loop=1, foreground_color=(0, 102, 204)):
            """
            Save a list of images as a GIF with looping set to 'loop' and stable black background.
            The foreground object will be set to the specified color while the background remains black.
            :param images: List of image arrays or PIL images.
            :param gif_path: File path for saving the GIF.
            :param duration: Duration for each frame in milliseconds.
            :param loop: Number of times the GIF loops. 0 = infinite.
            :param foreground_color: Color to set for the foreground (default is soft blue).
            """
            # Ensure images are numpy arrays and normalized for display
            time.sleep(2)
            image_list = []
            for i, img in enumerate(images):
                img = Image.fromarray(img)  # Convert to PIL Image if necessary
                # Convert to numpy array for processing
                img_array = np.array(img)
                img_array = cv2.resize(img_array,(150,150))
                # Ensure the image is in RGB format (3 channels)
                if img_array.ndim == 2:  # Grayscale image
                    img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB (3 channels)

                # Create a mask for the foreground (non-black pixels)
                foreground_mask = img_array.sum(axis=-1) != 0  # Identify non-black pixels (sum across RGB channels)

                # Apply the foreground color to non-black pixels
                img_array[foreground_mask] = foreground_color  # Set foreground color to the desired color

                # Convert the modified array back to an image
                # Ensure that the array has the correct dtype for the image (uint8)
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)  # Clip and convert to uint8

                img = Image.fromarray(img_array)

                image_list.append(img)

            # Save the images as a GIF
            imageio.mimsave(gif_path, image_list, duration=duration, loop=loop)
            print(f"GIF saved to {gif_path}")

        # Assuming 'images' is a list of frames
        save_gif(images, 'Predicted_pattern_.gif', duration=500, loop=1,
                 foreground_color=(112, 128, 144))  # Soft Blue foreground

        def display_gif_video(gif_path, window_name, duration=500):
            gif = imageio.mimread(gif_path)  # Read GIF file

            for frame in gif:
                # Convert the frame to an OpenCV-compatible format
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV compatibility

                # Display the frame in a separate OpenCV window
                cv2.imshow(window_name, frame)

                # Wait for a short duration before showing the next frame
                key = cv2.waitKey(duration)  # duration in milliseconds

                # If the user presses 'q', exit the loop
                if key == ord('q'):
                    break

        # display_gif_video('Predicted_pattern_.gif', window_name='Hidden Pattern', duration=500)

        def display_both_gifs(gif_path1, gif_path2, duration=500):
            # Display both GIFs simultaneously in separate windows
            window_name1 = 'GIF Playback 1'
            window_name2 = 'GIF Playback 2'

            # Start threads for both GIFs
            from threading import Thread

            # Create and start threads for both GIFs
            thread1 = Thread(target=display_gif_video, args=(gif_path1, window_name1, duration))
            thread2 = Thread(target=display_gif_video, args=(gif_path2, window_name2, duration))

            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()

            cv2.destroyAllWindows()  # Close the OpenCV windows after playback ends

        # # Main code to generate and display both GIFs at the same time
        # if __name__ == "__main__":
        #     # Assuming 'input_steregram' and 'Predicted_pattern_.gif' are the paths to the GIF files
        #     display_both_gifs(input_steregram, 'Predicted_pattern_.gif', duration=500)  # 500ms per frame
        self.gif2_path = 'Predicted_pattern_.gif'
        # self.frame1_ = tk.Label(self.root, text="GIF 2", bg="black", fg="white")
        # self.frame1_.place(x=380, y=20)
        if self.gif2_path:
            self.display_gif_frame(self.gif2_path, self.frame2)
            self.play_gifs()

    def display_gif_frame(self, gif_path, frame_label):
        gif = Image.open(gif_path)
        gif.seek(0)
        img = ImageTk.PhotoImage(gif)
        frame_label.config(image=img)
        frame_label.image = img

    def play_gifs(self):
        if self.gif1_path and self.gif2_path:
            threading.Thread(target=self.play_gif_thread, args=(self.gif1_path, self.frame1)).start()
            threading.Thread(target=self.play_gif_thread, args=(self.gif2_path, self.frame2)).start()
        else:
            print("Please load both GIFs first!")

    def play_gif_thread(self, gif_path, frame_label):
        gif = imageio.mimread(gif_path)
        while True:
            for frame in gif:
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(img)
                frame_label.config(image=img_tk)
                frame_label.image = img_tk
                time.sleep(0.5)  # Adjust frame delay

root = tk.Tk()
app = GIFPlayerApp(root)
root.mainloop()







