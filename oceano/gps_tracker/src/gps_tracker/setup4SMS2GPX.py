# Run the build process by entering 'setup.py py2exe' or
# 'python setup.py py2exe' in a console prompt.
#
# If everything works well, you should find a subdirectory named 'dist'
# containing exe files.


from distutils.core import setup

# import sys; sys.exit()
# sys.path.insert(0, <path_to_missing_modules>)

setup(
    # The first three parameters are not required, if at least a
    # 'version' is given, then a versioninfo resource is built from
    # them and added to the executables.
    version="0.0.1",
    description="convert SMS XML to GPX",
    name="converter from Andrey Korzh <andrey.korzh@atlantic.ocean.ru>",

    # targets to build
    console=["SMS2GPX.py"],
    #    options= {'py2exe':
    #        {'includes': ['lxml.etree', 'lxml._elementpath'], #, 'gzip'
    #        }
    # ,"skip_archive": True
    #   		}
    )
