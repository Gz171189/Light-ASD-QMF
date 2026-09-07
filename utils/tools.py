import os, subprocess, glob, pandas, tqdm, cv2, numpy, tempfile
from scipy.io import wavfile

def init_args(args):
    # The details for the following folders/files can be found in the annotation of the function 'preprocess_AVA' below
    args.modelSavePath    = os.path.join(args.savePath, 'model')
    args.scoreSavePath    = os.path.join(args.savePath, 'score.txt')
    args.trialPathAVA     = os.path.join(args.dataPathAVA, 'csv')
    args.audioOrigPathAVA = os.path.join(args.dataPathAVA, 'orig_audios')
    args.visualOrigPathAVA= os.path.join(args.dataPathAVA, 'orig_videos')
    args.audioPathAVA     = os.path.join(args.dataPathAVA, 'clips_audios')
    args.visualPathAVA    = os.path.join(args.dataPathAVA, 'clips_videos')
    args.trainTrialAVA    = os.path.join(args.trialPathAVA, 'train_loader.csv')

    if args.evalDataType == 'val':
        args.evalTrialAVA = os.path.join(args.trialPathAVA, 'val_loader.csv')
        args.evalOrig     = os.path.join(args.trialPathAVA, 'val_orig.csv')
        args.evalCsvSave  = os.path.join(args.savePath,     'val_res.csv')
    else:
        args.evalTrialAVA = os.path.join(args.trialPathAVA, 'test_loader.csv')
        args.evalOrig     = os.path.join(args.trialPathAVA, 'test_orig.csv')
        args.evalCsvSave  = os.path.join(args.savePath,     'test_res.csv')

    os.makedirs(args.modelSavePath, exist_ok = True)
    os.makedirs(args.dataPathAVA, exist_ok = True)
    return args


def preprocess_AVA(args):
    # This preprocesstion is modified based on this [repository](https://github.com/fuankarion/active-speakers-context).
    # The required space is 302 G.
    # If you do not have enough space, you can delate `orig_videos`(167G) when you get `clips_videos(85G)`.
    #                             also you can delate `orig_audios`(44G) when you get `clips_audios`(6.4G).
    # So the final space is less than 100G.
    # The AVA dataset will be saved in 'AVApath' folder like the following format:
    # ```
    # ├── clips_audios  (The audio clips cut from the original movies)
    # │   ├── test
    # │   ├── train
    # │   └── val
    # ├── clips_videos (The face clips cut from the original movies, be save in the image format, frame-by-frame)
    # │   ├── test
    # │   ├── train
    # │   └── val
    # ├── csv
    # │   ├── test_file_list.txt (name of the test videos)
    # │   ├── test_loader.csv (The csv file we generated to load data for testing)
    # │   ├── test_orig.csv (The combination of the given test csv files)
    # │   ├── train_loader.csv (The csv file we generated to load data for training)
    # │   ├── train_orig.csv (The combination of the given training csv files)
    # │   ├── trainval_file_list.txt (name of the train/val videos)
    # │   ├── val_loader.csv (The csv file we generated to load data for validation)
    # │   └── val_orig.csv (The combination of the given validation csv files)
    # ├── orig_audios (The original audios from the movies)
    # │   ├── test
    # │   └── trainval
    # └── orig_videos (The original movies)
    #     ├── test
    #     └── trainval
    # ```

    download_csv(args) # Take 1 minute
    download_videos(args) # Take 6 hours
    extract_audio(args) # Take 1 hour
    extract_audio_clips(args) # Take 3 minutes
    extract_video_clips(args) # Take about 2 days

def download_csv(args):
    # Take 1 minute to download the required csv files
    Link = "1C1cGxPHaJAl1NQ2i7IhRgWmdvsPhBCUy"
    csvTarGzPath = args.dataPathAVA + '/csv.tar.gz'

    # Check if tar.gz file already exists
    if not os.path.exists(csvTarGzPath):
        cmd = "gdown --id %s -O %s"%(Link, csvTarGzPath)
        subprocess.call(cmd, shell=True, stdout=None)

    cmd = "tar -xzvf %s -C %s"%(csvTarGzPath, args.dataPathAVA)
    subprocess.call(cmd, shell=True, stdout=None)
    os.remove(csvTarGzPath)

def download_videos(args):
    # Take 6 hours to download the original movies, follow this repository: https://github.com/cvdfoundation/ava-dataset
    for dataType in ['trainval', 'test']:
        fileList = open('%s/%s_file_list.txt'%(args.trialPathAVA, dataType)).read().splitlines()
        outFolder = '%s/%s'%(args.visualOrigPathAVA, dataType)
        for fileName in fileList:
            # Check if file already exists, skip if it does
            filePath = os.path.join(outFolder, fileName)
            if os.path.exists(filePath):
                print(f"File {fileName} already exists, skipping...")
                continue

            cmd = "wget -P %s https://s3.amazonaws.com/ava-dataset/%s/%s"%(outFolder, dataType, fileName)
            subprocess.call(cmd, shell=True, stdout=None)


def extract_audio(args):
    # Take 1 hour to extract the audio from movies
    for dataType in ['trainval', 'test']:
        inpFolder = '%s/%s' % (args.visualOrigPathAVA, dataType)
        outFolder = '%s/%s' % (args.audioOrigPathAVA, dataType)
        os.makedirs(outFolder, exist_ok=True)
        videos = glob.glob("%s/*" % (inpFolder))

        print(f"\nExtracting audio from {len(videos)} videos in {dataType}...")
        skipped = 0
        processed = 0

        for videoPath in tqdm.tqdm(videos, desc=f"{dataType}"):
            videoName = os.path.splitext(os.path.basename(videoPath))[0]
            audioPath = os.path.join(outFolder, videoName + '.wav')

            # 如果音频文件已存在且大小>0，跳过
            if os.path.exists(audioPath) and os.path.getsize(audioPath) > 0:
                skipped += 1
                continue

            context = _ava_context(dataType, videoName, 'N/A', 'N/A',
                                   video=videoPath, output=audioPath)
            # Publish only complete output: a failed ffmpeg must not leave a
            # partial .wav that a resumed run would mistake for finished work.
            with tempfile.TemporaryDirectory(dir=outFolder, prefix='.ava_audio_') as temporary:
                staged = os.path.join(temporary, 'audio.wav')
                cmd = ['ffmpeg', '-y', '-i', videoPath, '-async', '1', '-ac', '1',
                       '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-threads', '8',
                       staged, '-loglevel', 'error']
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
                    _require_ava_output(staged, context)
                    os.replace(staged, audioPath)
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise RuntimeError('Failed to extract AVA audio: {}\n{}'.format(
                        context, getattr(exc, 'stderr', str(exc)))) from exc
            processed += 1

        print(f"Completed {dataType}: processed={processed}, skipped_existing={skipped}, failed=0")


def _ava_context(dataType, video_id, entity_id, timestamp, **paths):
    return ('split={} video_id={} entity_id={} timestamp={} '.format(
        dataType, video_id, entity_id, timestamp) +
        ' '.join('{}={}'.format(key, value) for key, value in paths.items()))


def _require_ava_output(path, context):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError('Missing/empty required AVA output: {} path={}'.format(context, path))


def _existing_ava_output(path, context):
    if not os.path.exists(path):
        return False
    _require_ava_output(path, context)
    return True


def extract_audio_clips(args):
    # Take 3 minutes to extract the audio clips
    dic = {'train': 'trainval', 'val': 'trainval', 'test': 'test'}
    for dataType in ['train', 'val', 'test']:
        df = pandas.read_csv(os.path.join(args.trialPathAVA, '%s_orig.csv' % (dataType)), engine='python', header=0)
        dfNeg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        dfPos = df[df['label_id'] == 1]
        insNeg = dfNeg['instance_id'].unique().tolist()
        insPos = dfPos['instance_id'].unique().tolist()
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        entityList = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')
        audioFeatures = {}
        outDir = os.path.join(args.audioPathAVA, dataType)
        audioDir = os.path.join(args.audioOrigPathAVA, dic[dataType])

        for l in df['video_id'].unique().tolist():
            d = os.path.join(outDir, l[0])
            if not os.path.isdir(d):
                os.makedirs(d)

        print(f"\nExtracting audio clips for {dataType}: {len(entityList)} entities")
        skipped = 0
        processed = 0

        for entity in tqdm.tqdm(entityList, total=len(entityList), desc=f"{dataType}"):
            insData = df.get_group(entity)
            videoKey = insData.iloc[0]['video_id']
            start = insData.iloc[0]['frame_timestamp']
            end = insData.iloc[-1]['frame_timestamp']
            entityID = insData.iloc[0]['entity_id']
            insPath = os.path.join(outDir, videoKey, entityID + '.wav')
            audioFile = os.path.join(audioDir, videoKey + '.wav')
            context = _ava_context(dataType, videoKey, entityID, '{}..{}'.format(start, end),
                                   audio=audioFile, output=insPath)

            # 如果音频片段已存在且大小>0，跳过
            if _existing_ava_output(insPath, context):
                skipped += 1
                continue

            if videoKey not in audioFeatures.keys():
                if not os.path.isfile(audioFile):
                    raise FileNotFoundError('Missing original AVA audio: ' + context)
                try:
                    sr, audio = wavfile.read(audioFile)
                except Exception as exc:
                    raise RuntimeError('Failed to read AVA audio: ' + context) from exc
                if sr <= 0 or audio.size == 0:
                    raise RuntimeError('Invalid/empty AVA audio: ' + context)
                audioFeatures[videoKey] = (sr, audio)

            sr, audio = audioFeatures[videoKey]
            if not numpy.isfinite([start, end]).all():
                raise RuntimeError('Invalid AVA audio timestamps: ' + context)
            audioStart = int(float(start) * sr)
            audioEnd = int(float(end) * sr)
            if not 0 <= audioStart < audioEnd <= len(audio):
                raise RuntimeError('Empty/out-of-range AVA audio clip: ' + context)
            audioData = audio[audioStart:audioEnd]
            try:
                with tempfile.TemporaryDirectory(dir=os.path.dirname(insPath),
                                                 prefix='.ava_clip_') as temporary:
                    staged = os.path.join(temporary, 'clip.wav')
                    wavfile.write(staged, sr, audioData)
                    _require_ava_output(staged, context)
                    os.replace(staged, insPath)
            except Exception as exc:
                raise RuntimeError('Failed to write AVA audio clip: ' + context) from exc
            _require_ava_output(insPath, context)
            processed += 1

        if processed + skipped != len(entityList):
            raise RuntimeError('Incomplete AVA audio split={}'.format(dataType))
        print(f"Completed {dataType}: processed={processed}, skipped_existing={skipped}, failed=0")


def extract_video_clips(args):
    # Take about 2 days to crop the face clips.
    # You can optimize this code to save time, while this process is one-time.
    # If you do not need the data for the test set, you can only deal with the train and val part. That will take 1 day.
    dic = {'train': 'trainval', 'val': 'trainval', 'test': 'test'}
    for dataType in ['train', 'val', 'test']:
        df = pandas.read_csv(os.path.join(args.trialPathAVA, '%s_orig.csv' % (dataType)), engine='python', sep=',',
                             encoding='utf-8', header=0)
        dfNeg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        dfPos = df[df['label_id'] == 1]
        insNeg = dfNeg['instance_id'].unique().tolist()
        insPos = dfPos['instance_id'].unique().tolist()
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        entityList = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')
        outDir = os.path.join(args.visualPathAVA, dataType)
        audioDir = os.path.join(args.visualOrigPathAVA, dic[dataType])

        # 预先创建所有需要的目录
        for l in df['video_id'].unique().tolist():
            d = os.path.join(outDir, l[0])
            if not os.path.isdir(d):
                os.makedirs(d)

        # 缓存视频文件路径和VideoCapture对象
        videoFileCache = {}
        currentVideo = None
        currentVideoKey = None

        print(f"\nProcessing {dataType} set: {len(entityList)} entities")
        processed = skipped = expected = 0
        try:
            for entity in tqdm.tqdm(entityList, total=len(entityList), desc=f"{dataType}"):
                insData = df.get_group(entity)
                videoKey = insData.iloc[0]['video_id']
                entityID = insData.iloc[0]['entity_id']
                insDir = os.path.join(outDir, videoKey, entityID)
                expected += len(insData)

                # Check each target before opening the input video. Fully and
                # partially completed entities remain resumable.
                for _, row in insData.iterrows():
                    timestamp = row['frame_timestamp']
                    imageFilename = os.path.join(insDir, ("%.2f" % timestamp) + '.jpg')
                    videoPattern = os.path.join(audioDir, '{}.*'.format(videoKey))
                    context = _ava_context(dataType, videoKey, entityID, timestamp,
                                           video=videoFileCache.get(videoKey, videoPattern),
                                           output=imageFilename)
                    if _existing_ava_output(imageFilename, context):
                        skipped += 1
                        continue
                    os.makedirs(insDir, exist_ok=True)

                    if videoKey != currentVideoKey:
                        if currentVideo is not None:
                            currentVideo.release()
                            currentVideo = None
                        if videoKey not in videoFileCache:
                            matches = glob.glob(videoPattern)
                            if not matches:
                                raise FileNotFoundError('Missing original AVA video: ' + context)
                            videoFileCache[videoKey] = matches[0]
                        try:
                            currentVideo = cv2.VideoCapture(videoFileCache[videoKey])
                        except cv2.error as exc:
                            raise RuntimeError('Failed to open AVA video: ' + context) from exc
                        currentVideoKey = videoKey
                    context = _ava_context(dataType, videoKey, entityID, timestamp,
                                           video=videoFileCache[videoKey], output=imageFilename)
                    if not currentVideo.isOpened():
                        raise RuntimeError('Failed to open AVA video: ' + context)
                    if not numpy.isfinite(timestamp) or timestamp < 0:
                        raise RuntimeError('Invalid AVA frame timestamp: ' + context)
                    try:
                        if not currentVideo.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1e3):
                            raise RuntimeError('Failed to seek AVA video: ' + context)
                        ret, frame = currentVideo.read()
                    except cv2.error as exc:
                        raise RuntimeError('Failed to read AVA frame: ' + context) from exc
                    if (not ret or frame is None or frame.size == 0 or
                            frame.ndim != 3 or frame.shape[2] != 3):
                        raise RuntimeError('Failed to read AVA frame: ' + context)

                    h, w = frame.shape[:2]
                    box = [row['entity_box_x1'], row['entity_box_y1'],
                           row['entity_box_x2'], row['entity_box_y2']]
                    if not numpy.isfinite(box).all():
                        raise RuntimeError('Invalid AVA face box: ' + context)
                    x1, y1 = int(box[0] * w), int(box[1] * h)
                    x2, y2 = int(box[2] * w), int(box[3] * h)
                    # Preserve the existing clipping and crop pixel values.
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        raise RuntimeError('Invalid AVA face crop: ' + context)
                    face = frame[y1:y2, x1:x2, :]
                    if face.size == 0:
                        raise RuntimeError('Empty AVA face crop: ' + context)
                    try:
                        with tempfile.TemporaryDirectory(dir=insDir, prefix='.ava_frame_') as temporary:
                            staged = os.path.join(temporary, 'frame.jpg')
                            if not cv2.imwrite(staged, face):
                                raise RuntimeError('cv2.imwrite returned False')
                            _require_ava_output(staged, context)
                            os.replace(staged, imageFilename)
                    except Exception as exc:
                        raise RuntimeError('Failed to write AVA face crop: ' + context) from exc
                    _require_ava_output(imageFilename, context)
                    processed += 1
        finally:
            if currentVideo is not None:
                currentVideo.release()

        if processed + skipped != expected:
            raise RuntimeError('Incomplete AVA video split={}'.format(dataType))
        print(f"Completed {dataType}: processed={processed}, skipped_existing={skipped}, failed=0")
