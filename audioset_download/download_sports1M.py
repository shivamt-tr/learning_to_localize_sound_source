import time
import random
import os
from tqdm import tqdm
from youtube_dl import YoutubeDL
from multiprocessing import Process


def download_video(start,end):
  path = '/backup/data3/shivam/audio-visual-dataset/unbalanced_train_split_6.csv'

  urls = []
  filenames = []
  starts, ends = [], []
  counter = 0
  with open(path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

    ytid, start1, end1, label = row
    filename = f"{ytid}_{label}"
    filenames.append(filename)

    # filepath = Path(outdir) / filename

    url = f'http://www.youtube.com/watch?v={ytid}'
    urls.append(url)
    starts.append(start1)
    ends.append(end1)


  for index in tqdm(range(start,end)):
    try:
      # for x, y, z in os.walk('.'):
      #   pass
      # file_names = []
      # for file in z:
      #   if '.mp4' in file:
      #     file_names.append(file.split('.mp4')[0][:file.rindex('-')])
      #print(line)
      url = urls[index]
      print(url)
      #if counter == 2:
       # break
      youtube_dl_opts = {}
      wait = random.randint(0, 2)
      print(wait)
      time.sleep(wait)
      with YoutubeDL(youtube_dl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_title = info_dict.get('title', None)
        print(f'video_title {video_title}')
      # if video_title in file_names:
      #   print('exist in folder')
        continue
      os.system(f'youtube-dl --recode-video mp4 {url} --output {filenames[index]}')
      os.system(f"ffmpeg -ss '{starts[index]}' -i {filenames[index]}.mp4 -t '{ends[index]-starts[index]+1}' -f mp4 -threads 1 -c:v copy -c:a copy {filenames[index]}_output.mp4")
      os.system(f'rm {filenames[index]}.mp4')
      #counter += 1
    except KeyboardInterrupt:
      exit(0)
    except:
      pass

if __name__ == '__main__':
    processes = []
    batch_start = 0
    for i in range(12):
      process = Process(target=download_video,args=(batch_start,batch_start+1000,))
      processes.append(process)
      batch_start+=1000

    for process in processes:
      process.start()
      time.sleep(5)

    for process in processes:
      process.join()
