from subprocess import CalledProcessError  #, check_output, Popen, PIPE
import asyncio
from pathlib import Path


duration = 0.25
glob = '*.png'

filenames = list(Path().glob(glob))
# path = str(Path("").absolute()).replace("\\", "/")
print(len(filenames), glob, 'files found')

if not Path("ffmpeg_in.txt").is_file():
    # Write input files list with duration settings
    with Path("ffmpeg_in.txt").open("w") as outfile:
        for filename in filenames:
            outfile.write(f'file {filename}\nduration {duration}\n')
        # Due to a quirk, the last image has to be specified twice - the 2nd time without any duration directive
        outfile.write(f'file {filename}')

ffmpeg_path = r'c:\Programs\_multimedia\_video\_libs\ffmpeg-win64\bin\ffmpeg.exe'




#
# c:\Programs\_multimedia\_video\_libs\ffmpeg-win64\bin\ffmpeg.exe -f concat -safe 0 -i ffmpeg_in.txt -vf scale=3204:2442 -c:v libx264 -crf 23 -pix_fmt yuv420p output.mp4

#
# ffmpeg -f concat -safe 0 -i ffmpeg_in.txt -vf scale=3204:2442 -c:v wmv2 -b:v 1500k output.wmv

# ffmpeg.exe -f concat -safe 0 -i ffmpeg_in.txt -c:v mpeg2video -b:v 10M -maxrate 15M -bufsize 15M -qmin 1 -qmax 5 -g 30 -bf 2 -trellis 2 -cmp 2 -subcmp 2 -pix_fmt yuv420p output.mpg

# Best for compatibility, worst for size:
# c:\Programs\_multimedia\_video\_libs\ffmpeg-win64\bin\ffmpeg.exe -f concat -safe 0 -i ffmpeg_in.txt -c:v msmpeg4v2 -qscale:v 3 -pix_fmt yuv420p output.avi
ext, cmd = "avi", ['-c:v', 'msmpeg4v2', '-qscale:v', '3']

out_file = f'''{filenames[0].stem.replace(',', '')}.{ext}'''
command_line = ([
    ffmpeg_path,
    '-r', '4', '-hwaccel', 'd3d11va',  #
    '-f', 'concat', '-safe', '0', '-i', 'ffmpeg_in.txt',

    # # crop & subtitles
    # '-vf', ','.join([
    #     'crop=2974:2422:3631:2315',  # 3476:3952:71:1304',
    #     "subtitles=subtitles.srt"  # :force_style='Fontname=DejaVu Serif,PrimaryColour=&HCCFF0000'
    #     # ,format=yuv420p
    # ]),

    # # crop (Picture width must be an integer multiple of the specified chroma subsampling)
    # '-vf', ','.join([
    #     'crop=3204:2442:1:0',  # 3476:3952:71:1304',
    # ]),
    # '-c:v', 'libx265', '-pix_fmt', 'yuv420p', '-tune', 'animation',

    # , '-crf', '28' , '-r', '60',
    # '-x264opts', 'opencl' - gets error
    # 'libx264' - gets artifacts
    ]
    + cmd
    + ['-pix_fmt', 'yuv420p', out_file]
)
print(' '.join(command_line))


async def main():
    process = None
    try:
        process = await asyncio.create_subprocess_exec(*command_line)
        print(f'subprocess ({process}) started')
        await process.wait()
        if process.returncode:
            raise BaseException(f'Error code: {process.returncode}')
        # # output = check_output(command_line, shell=True)
        # pipe = Popen(command_line, shell=True, stdin=PIPE, stdout=PIPE).stdout
        # output = pipe.read().decode()
        # pipe.close()
    # except CalledProcessError:
    #     print('Failed')
    except Exception as e:
        print('Failed:', e)
        if process:
            process.terminate()
    else:
        if process.stdout:
            print('output:', process.stdout)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Ok>")


# entry point
asyncio.run(main())
