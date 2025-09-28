# @+leo-ver=5-thin
# @+node:korzh.20180530093423.1: * @file utils4work.py
# @+others
# @+node:korzh.20180530001926.1: ** run_module decorated
# @@language python
# @@first

# from datetime import datetime; datetime.now.strftime('%d.%m.%Y %H:%M:%S')
g.es(g.time.strftime('%d.%m.%Y %H:%M:%S'), color='green')

import importlib
# @+<<imports>>
# @+node:korzh.20180530001926.3: *3* <<imports>>
# @+<<imports of run_with_args>>
# @+node:korzh.20180530001926.4: *4* <<imports of run_with_args>>
# @-<<imports of run_with_args>>
# @+<<imports of redirect_to_stdout>>
# @+node:korzh.20180530001926.5: *4* <<imports of redirect_to_stdout>>
import io
# @-<<imports of redirect_to_stdout>>
# @+<<imports of run_module_main>>
# @+node:korzh.20180530001926.6: *4* <<imports of run_module_main>>
import sys
from contextlib import redirect_stdout
from functools import wraps


# @-<<imports of run_module_main>>
# @-<<imports>>
# @+others
# @+node:korzh.20180530001926.2: *3* clearing the Log window
# @+at
# When developing scripts that use Log window to display results, it is
# sometimes useful to clear Log window by inserting the following two lines
# at the beginning of your script::
# 
# c.frame.log.selectTab('Log')
# c.frame.log.clearLog()
# @+node:korzh.20180530001926.7: *3* redirect_to_stdout_decor
# @@language python

def redirect_to_stdout_decor(*args):
    def redirect_stdout_generator(old_func):
        @wraps(old_func)
        def _func(*args):
            with io.StringIO() as buf, redirect_stdout(buf):
                old_func(*args)
                print('....................')
                g.es(buf.getvalue(), color='#00AA00')

        return _func

    return redirect_stdout_generator


# @+node:korzh.20180530001926.8: *3* run_module_main def
def run_module_main(module, fun='main'):
    mod = importlib.import_module(module)

    func = getattr(mod, fun)
    return func()


# @+node:korzh.20180530001926.9: *3* cmd(func, command_line_args)
# @@language python

def cmd(func, command_line_args):
    '''
    Assigns sys.argv[1:] to command_line_args and calls func()
    Useful for functions which analyses command line arguments
        
    func:              function, which can be called without any parameters
    command_line_args: sequence of command line arguments 
    '''
    # g.trace(command_line_args)
    # if isinstance(sys.argv, list):
    # sys.argv = sys.argv[:1]             # keep only name of current module
    # g.trace(sys.argv)
    # sys.argv.extend(command_line_args)  # sys.argv.append(*command_line_args)
    # g.trace(sys.argv)
    # g.pdb()

    # else:
    # g.pdb()

    sys.argv[1:] = list(command_line_args)

    return func()


# @-others
# run_module decorated
@redirect_to_stdout_decor()
def run_module(module, arg):
    def run_modul():
        run_module_main(module)

    cmd(run_modul, arg)


# @+node:korzh.20180528132430.1: ** colormap_matplotlib2surfer.py
def colormap_matplotlib2surfer(palette_name='jet', save_dir='/mnt/D/workData/~pattern~/CTD'):
    """
    Saves matplotlib palette to Golden Software's Surfer ".clr" file named same as palette
    :param palette_name: str, matplotlib palette name
    :save_dir: str, directory where to save <palette_name>.clr
    """
    cMap = plt.cm.get_cmap(palette_name)

    x_list = sort(list(set(vstack((
        cMap._segmentdata['red'],
        cMap._segmentdata['green'],
        cMap._segmentdata['blue'])
        )[:, 0:2].ravel())))

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    rgba = [mapper.to_rgba(x, bytes=True) for x in x_list]
    out = c_[np.atleast_2d(x_list).T * 100, np.array(rgba)]

    np.savetxt(os.pathh.join(save_dir, '{}.clr'.format(palette_name)),
               out, header='ColorMap 2 1', delimiter='\t', fmt='%3.1f\t%3.0f\t%3.0f\t%3.0f\t%3.0f', comments='')
# @-others
# @-leo
