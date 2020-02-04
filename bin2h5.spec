# -*- mode: python -*-

block_cipher = None


a = Analysis(['d:/Work/_Python3/And0K/h5toGrid/to_pandas_hdf5/bin2h5.py'],
             pathex=[
             'd:/Work/_Python3/And0K/h5toGrid', 
             'd:/WorkCode/Miniconda3/Scripts',
             #'C:/PROGRA~2/WI3CF2~1/10/Redist/ucrt/DLLs/x86'
             ],
             datas=[#('d:/Work/_Python3/And0K/scraper3/scrapwww/README.rst', '.'),
	('d:/Work/_Python3/And0K/h5toGrid/to_pandas_hdf5/bin2h5_ini/bin_Brown.ini', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='bin2h5',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='bin2h5')
