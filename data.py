from pydub import AudioSegment
from pyAudioAnalysis import audioBasicIO as pyab
from pyAudioAnalysis import audioFeatureExtraction as pyaf
from pyAudioAnalysis import audioSegmentation as pyas
import os
import pickle
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def save_file(filename, musicData, kind='pkl'):

    if kind == 'pkl':
        save_loc = os.curdir + '/' + filename + '.pkl'

        with open(save_loc, "wb") as f:
            pickle.dump(musicData, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    elif kind == 'pq':
        assert isinstance(musicData, pd.DataFrame)
        table = pa.Table.from_pandas(musicData)
        pq.write_table(table, os.curdir+'/'+filename+'.parquet')


def getData(filePath):

    kind = filePath.split('.')[-1]

    if kind == 'pkl':
        with open(filePath, 'rb') as f:
            musicData = pickle.load(f)
            f.close()

    elif kind == 'parquet':
        table = pq.read_table(filePath)
        musicData = table.to_pandas()

    return musicData


def createData(dirPath=os.curdir + '/music/', save_loc=None):

    musicData = []
    files = [x for x in os.listdir(dirPath) if x[0] != '.']

    for file in files:
        print(f'Tracks Added: {str(len(musicData))} / {str(len(files))}\rCurrently Loading "{file}" ...\n')
        music = extractFeatures(dirPath, file)
        musicData.append(music)

    if save_loc:
        save_file(save_loc)

    return musicData


def extractFeatures(filePath, fileName, music_genre=None, window=0.05, step=0.05, thumbnailSize=20, playDuration=20):

    music = {}
    music["fileName"] = fileName
    music["filePath"] = filePath
    music["genre"] = music_genre
    #music["fileInfo"]=[]
    [fs, x] = pyab.readAudioFile(filePath+fileName)

    if thumbnailSize is not None:
        [A1, A2, B1, B2, S] = pyas.musicThumbnailing(x, fs, thumbnailSize)
        B = A1
        E = B+playDuration
        audio = AudioSegment.from_file(filePath+fileName)
        music["audio"] = audio[int(1000*B):int(1000*E)]
    else:
        B = 0
        E = B+playDuration
        audio = AudioSegment.from_file(filePath+fileName)
        music["audio"] = audio[int(1000*B):int(1000*E)]

    x = pyab.stereo2mono(x)
    x = x[int(fs*B):int(fs*E)]
    music["thumbNail"] = (B, E)
    music["fileData"] = x
    music["stFeatures"] = pyaf.stFeatureExtraction(x, fs, window*fs, step*fs)
    music["FrameRate"] = fs
    #music["classifierResult"] = None
    #music["UserResult"] = None

    return music

    # windowDuration = window * fs
    # total duration =  x.shape[0]/fs
    # Data Points = int(x.shape[0]/(step*fs))

