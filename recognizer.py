from Tkinter import *
from record import record_to_file
from features import mfcc, logfbank
import scipy.io.wavfile as wav

class Application(Frame):

    def createWidgets(self):
        self.button_image = PhotoImage(file="button.gif")
        self.RECORD = Button(self, image=self.button_image, width="100", height="100", command=record_and_test)
        self.RECORD.pack({"side" : "left"})

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

def record_and_test():
    record_to_file('demo.wav')
    (rate,sig) = wav.read('demo.wav')
    # TODO Feed into ANN

if __name__ == '__main__':

    # Display GUI
    root = Tk()
    app = Application(master=root)
    app.mainloop()
    root.destroy()