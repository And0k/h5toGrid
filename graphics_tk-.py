
import threading
import time
import queue
import functools

# based on https://stackoverflow.com/a/55268663/2028147

plt = None
ax = None
fig = None

#ript(Run In Plotting Thread) decorator
def ript(function):
    def ript_this(*args, **kwargs):
        global send_queue, return_queue, plot_thread
        if threading.currentThread() == plot_thread: #if called from the plotting thread -> execute
            return function(*args, **kwargs)
        else: #if called from a diffrent thread -> send function to queue
            send_queue.put(functools.partial(function, *args, **kwargs))
            return_parameters = return_queue.get(True) # blocking (wait for return value)
            return return_parameters
    return ript_this

# list functions in matplotlib you will use

#functions_to_decorate = [[matplotlib.axes.Axes,'plot'],
#                          [matplotlib.figure.Figure,'savefig'],
#                          [matplotlib.backends.backend_tkagg.FigureCanvasTkAgg,'draw'],
#                         ]

# #add the decorator to the functions
# for obj, function in functions_to_decorate:
#     setattr(obj, function, ript(getattr(obj, function)))

# function that checks the send_queue and executes any functions found
def update_figure(window, send_queue, return_queue):
    try:
        callback = send_queue.get(False)  # get function from queue, false=doesn't block
        return_parameters = callback() # run function from queue
        return_queue.put(return_parameters)
    except:
        pass
    window.after(10, update_figure, window, send_queue, return_queue)

# function to start plot thread
def plot_in_TkAgg():
    # we use these global variables because we need to access them from within the decorator
    global plot_thread, send_queue, return_queue
    return_queue = queue.Queue()
    send_queue = queue.Queue()
    plot_thread=threading.currentThread()
    # we use these global variables because we need to access them from the main thread
    global ax, fig

    import matplotlib
    matplotlib.use('TkAgg', force=True)

    # try:
    #         #'Qt5Agg')  # must be before importing plt (rases error after although documentation sed no effect)
    #     matplotlib.interactive(False)
    # except ImportError:
    #     print('matplotlib can not import Qt5Agg backend - may be errors in potting')
    #     pass
    from matplotlib import pyplot as plt

    if matplotlib.get_backend() != 'TkAgg':
        plt.switch_backend('TkAgg')

    fig, ax = plt.subplots()
    # we need the matplotlib window in order to access the main loop
    window=plt.get_current_fig_manager().window
    # we use window.after to check the queue periodically
    window.after(10, update_figure, window, send_queue, return_queue)
    # we start the main loop with plt.plot()
    plt.show()


def plot_bad_time_in_thread(*args, **kwargs):
    #start the plot and open the window
    try:
        thread = threading.Thread(target=plot_in_TkAgg)
    except ImportError:
        l.warning('Can not load TkAgg to draw outcide of main thread')
        return ()

    thread.setDaemon(True)
    thread.start()
    time.sleep(1) #we need the other thread to set 'fig' and 'ax' before we continue

    #run the simulation and add things to the plot
    #global ax, fig

    plot_bad_time(*args, **kwargs)

    # for i in range(10):
    #     ax.plot([1,i+1], [1,(i+1)**0.5])
    #     fig.canvas.draw()
    #     fig.savefig('updated_figure.png')
    time.sleep(1)
    print('Done')
    thread.join() #wait for user to close window


@ript
def plot_bad_time(b_ok, cfg_in, idel, msg, path_save_image, t, tim, tim_range: Optional[Tuple[Any,Any]]=None):

    global plt

    plt.figure('Decreasing time corr')
    plt.title(msg)
    plt.plot(idel, tim.iloc[idel], '.m')
    plt.plot(np.arange(t.size), tim, 'r')
    plt.plot(np.flatnonzero(b_ok), pd.to_datetime(t[b_ok], utc=True), color='g', alpha=0.8)
    if 'path' in cfg_in and path_save_image:
        if not PurePath(path_save_image).is_absolute():
            path_save_image = PurePath(cfg_in['path']).with_name(path_save_image)
            dir_create_if_need(path_save_image)
        fig_name = path_save_image / '{:%y%m%d_%H%M}-{:%H%M}.png'.format(
            *tim_range if tim_range[0] else tim.iloc[[0, -1]])
        plt.savefig(fig_name)
        l.info(' - figure saved to %s', fig_name)
    # plt.show(block=True)
