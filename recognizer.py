from Tkinter import *
from threading import Thread
from record import record_to_file
from features import mfcc, logfbank
import scipy.io.wavfile as wav

class Application(Frame):

    def createWidgets(self):
        self.button_image = PhotoImage(file="button.gif")
        self.RECORD = Button(self, image=self.button_image, width="100", height="100", command=self.record_and_test)
        self.RECORD.pack()
        self.TEXTBOX = Text(self, height="1", width="15")
        self.TEXTBOX.pack()

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()
        self.TEXTBOX.insert(INSERT, "Press to record")
        self.TEXTBOX.tag_config("recording", foreground="red", justify="center")
        self.TEXTBOX.tag_config("success", foreground="darkgreen", justify="center")
        self.TEXTBOX.configure(state="disabled")

    def record_and_test(self):
        self.TEXTBOX.configure(state="normal")
        self.TEXTBOX.delete("1.0", END)
        self.TEXTBOX.insert(INSERT, "Recording")
        self.TEXTBOX.tag_add("recording", "1.0", END)
        self.TEXTBOX.configure(state="disabled")
        self.master.update()

        recorder_thread = Thread(target = record_to_file, args=('test.wav',))
        recorder_thread.start()
        recorder_thread.join()
        (rate,sig) = wav.read('test.wav')
        # TODO Feed into ANN

        self.TEXTBOX.configure(state="normal")
        self.TEXTBOX.delete("1.0", END)
        self.TEXTBOX.tag_remove("recording", "1.0")
        self.TEXTBOX.insert(INSERT, "Completed")
        self.TEXTBOX.tag_add("success", "1.0", END)
        self.TEXTBOX.configure(state="disabled")

def record(filename="test.wav"):
    pass

if __name__ == '__main__':

    # Display GUI
    root = Tk()
    app = Application(master=root)
    app.mainloop()
    #root.destroy()