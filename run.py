import radiob
import sys
from tkinter import Tk
#sys.path.append('dicom_vol_viewer')
#if __name__ == "__main__":
gui_obj = Tk()
a = radiob.GUI(gui_obj,'../TAVR_MAIN')
gui_obj.mainloop()