from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox, filedialog
import os.path
import errno
import shutil
import app

root = Tk()         #root window
e = Entry(root, width=35, borderwidth=5)
root.resizable(False, False)            #disable resizing of root window

# Functions
def browse_button():
    root.filename = filedialog.askopenfilename(initialdir=".", title="Select a video", filetypes=(("mp4 files", "*.mp4"),("avi files", "*.avi")))           #selected video (file)

    if os.path.exists(root.filename):           #if the video (file) exists
        browse_label = Label(root, text=root.filename + '\n').pack()            #path of the selected video (file)
        text_label = Label(root, text="If you want, you can change the file name before START:").pack()
        e.pack()
        startButton = Button(root, text="Start", command=button_start, pady=8, fg="white", bg="black").pack()           #Start Button

def button_start():
    k = 1
    x = root.filename
    top = Toplevel()
    make_sure_path_exists(os.path.abspath(os.path.dirname(__file__)) + "\\PV")
    input_file = None
    while True:
        if e.get() != "":
            input_file = e.get()
            shutil.copy(x , os.path.abspath(os.path.dirname(__file__)) + "\\PV\\" + e.get() + ".mp4")          #copies the selected file (as we wish)
            break
        elif not os.path.isfile(os.path.abspath(os.path.dirname(__file__)) + "\\PV\\" + str(k) + ".mp4"):
            shutil.copy(x , os.path.abspath(os.path.dirname(__file__)) + "\\PV\\" + str(k) + ".mp4")
            break
        else:
            k = k + 1
            continue
        
    #---The processing of the Video---

    my_label2 = Label(top, text="Your processed video is in the PVs folder.").pack()
    print(input_file)
    btn = Button(top, text="Close", command=top.destroy).pack()         #Close Button (after clicking Start Button)

def make_sure_path_exists(path):            #checks if path (folder/file) exists, if not then it will be created
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

# Main
def main():
    root.title('License Plate Recognition')
    myLabel1 = Label(root, text="Welcome to the License Plate Recognition Program!").pack()

    my_img = ImageTk.PhotoImage(Image.open(os.path.abspath(os.path.dirname(__file__)) + "\\resources\\1.jpg"))
    my_label = Label(image=my_img).pack()

    uploadButton = Button(root, text="Browse a video", command=browse_button, pady=3, fg="white", bg="blue")
    uploadButton.pack()

    root.mainloop()

if __name__ == "__main__":
    main()