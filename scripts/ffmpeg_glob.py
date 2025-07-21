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

out_file = f'''{filenames[0].stem.replace(',', '')}.mp4'''

command_line = ([
    ffmpeg_path, '-r', '4', '-hwaccel', 'd3d11va',  #
    '-f', 'concat', '-safe', '0', '-i', 'ffmpeg_in.txt',
    '-vf', ','.join([
        'crop=2974:2422:3631:2315',  # 3476:3952:71:1304',
        "subtitles=subtitles.srt"  # :force_style='Fontname=DejaVu Serif,PrimaryColour=&HCCFF0000'
        # ,format=yuv420p
    ]),
    '-c:v', 'libx265', '-pix_fmt', 'yuv420p', '-tune', 'animation',
    # , '-crf', '28' , '-r', '60',
    # '-x264opts', 'opencl' - gets error
    # 'libx264' - gets artifacts
    out_file
])
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
        print('ok>')


# entry point
asyncio.run(main())
