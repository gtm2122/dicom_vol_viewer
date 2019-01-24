import gdcm
import sys
import pydicom
#import matplotlib.pyplot as plt
import os

#if __name__ == "__main__":

#def decomp_img(file1,file2):

#returns the decompressed pixel_array for displaying the dicom
# if os.path.isdir(file2):
#    temp_files
import shutil

def decomp(file1,file2='temp_file_22.dcm',return_dcm=False):
    #file1 = '/data/gabriel/TAVR/TAVR_Sample_Study/IM-0003-1528.dcm'#sys.argv[1] # input filename

    #file2 = file2#sys.argv[2] # output filename
    if os.path.isfile(file2):
        os.remove(file2)
    reader = gdcm.ImageReader()
    reader.SetFileName( file1 )
    reader.GetFile()
    if not reader.Read():
        sys.exit(1)

    change = gdcm.ImageChangeTransferSyntax()
    change.SetTransferSyntax( gdcm.TransferSyntax(gdcm.TransferSyntax.ImplicitVRLittleEndian) )
    change.SetInput( reader.GetImage() )
    if not change.Change():
        sys.exit(1)

    writer = gdcm.ImageWriter()
    writer.SetFileName( file2 )
    writer.SetFile( reader.GetFile() )
    writer.SetImage( change.GetOutput() )
    if not writer.Write():
        sys.exit(1)
    
    if return_dcm:
        return pydicom.dcmread(file2,force=True)
    else:
        return pydicom.dcmread(file2,force=True).pixel_array

if __name__=="__main__":
    decomp(file1 = '/data/gabriel/TAVR/TAVR_Sample_Study/IM-0003-1528.dcm',file2='temp_file_22.dcm')
# if not writer.Write():
#     sys.exit(1)
#return pydicom.dcmread(file2,force=True).pixel_array


    