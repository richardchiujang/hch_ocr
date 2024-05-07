import compileall
import glob
import shutil
import os 

compileall.compile_dir('./', force=True)

glist = glob.glob('**/*.cpython-310.pyc', recursive=True)
for l in glist:
    print(l, l.replace('.cpython-310.pyc', '.pyc').replace('__pycache__\\', ''))
    shutil.move(l, l.replace('.cpython-310.pyc', '.pyc').replace('__pycache__\\', ''))


plist = glob.glob('./**/*.py', recursive=True)  
ex_list = ['.\py_to_pyc.py', '.\hch_ocr.py']

# ###########################################################
# ## caution!!! Remove all .py files except the current file
# ###########################################################
# for l in plist:
#     if l not in ex_list:
#         print(l)
#         os.remove(l)



    
