#from cx_Freeze import setup, Executable
from distutils.core import setup
import py2exe
# import os.path
# #import matplotlib
# import scipy
# import radiob
# if __name__=="__main__":
#     scipy_p = os.path.dirname(scipy.__file__)
#     additional_mods = ['numpy.core._methods','numpy.matlib','pandas.core'] 
#     PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
#     pic_dir = './welcome.png'
#     anno_p = './AT2'

#     build_exe_options = {"includes":  additional_mods,
#                          "include_files":[scipy_p,anno_p,pic_dir],
#                          "excludes":['scipy.spatial.cKDTree'],
#                         }
#     dep = [i for i in os.listdir(os.path.join(PYTHON_INSTALL_DIR,'Library','bin')) if 'mkl' in i]
#     dep.append('libiomp5md.dll')
#     #dependencies = ['libiomp5md.dll', 'mkl_core.dll', 'mkl_def.dll', 'mkl_intel_thread.dll']
#     PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
#     DLLS_FOLDER = os.path.join(PYTHON_INSTALL_DIR,'Library','bin')

#     for d in dep:
#         build_exe_options['include_files'].append(os.path.join(DLLS_FOLDER,d))

#     os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
#     os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

setup(name="radiob",
      version="0.2",
      description="",
      options={
          'build_exe':build_exe_options
          },
      executables=[Executable("radiob.py")]
     )

import os
#os.makedirs('build\\TAVR_MAIN\\TAVR_ROOT')
