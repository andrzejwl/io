from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox, filedialog
import os.path

# Functions
def browse_button():
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select a video", filetypes=(("mp4 files", "*.mp4"),("avi files", "*.avi")))

    if os.path.exists(root.filename):
        browse_label = Label(root, text=root.filename).pack()
        startButton = Button(root, text="Start", command=button_start, pady=5, fg="white", bg="black").pack()
        e = Entry(root, width=35, borderwidth=5)
        e.pack()

def button_start():
    top = Toplevel()
    my_label2 = Label(top, text="Your processed video is in the PVs folder.").pack()
    btn = Button(top, text="Close", command=top.destroy).pack()

# Main
root = Tk()
root.title('License Plate Recognition')
myLabel1 = Label(root, text="Welcome to the License Plate Recognition Program!").pack()

my_img = ImageTk.PhotoImage(Image.open("C:/Users/krzys/Desktop/1.jpg"))
my_label = Label(image=my_img).pack()

uploadButton = Button(root, text="Browse a video", command=browse_button, pady=3, fg="white", bg="blue")
uploadButton.pack()

root.mainloop()