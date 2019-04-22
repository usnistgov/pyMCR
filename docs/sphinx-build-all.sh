./make.bat clean

# ../setup.py is the EXCLUSION path
sphinx-apidoc.exe -f -o ./source/ .. ../setup.py

./make.bat html