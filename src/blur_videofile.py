import argparse
import datetime
import os
from os import path
import shlex
import shutil
import subprocess
import time
import tempfile

import cv2
import numpy as np
import face_recognition
import tensorflow as tf


logging = tf.logging
logging.set_verbosity(logging.INFO)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        '-f',
        help='Path to video file.'
    )
    parser.add_argument(
        '--models-dir',
        '-m',
        default=os.getenv('DATA_DIR'),
        help='Face recognition models directory.'
    )
    parser.add_argument(
        '--output',
        '-O',
        help='Output file location',
        default='output.mp4'
    )
    parser.add_argument(
        '--train-dir',
        help='Output file location',
        default=os.getenv('TRAINING_DIR')
    )
    parser.add_argument(
        '--model',
        help='Model to use',
        choices=['hog', 'cnn'],
        default='hog',
    )
    parser.add_argument(
        '--log-images',
        help='How much images should be logged in tensorboard for this video.',
        metavar='<int>',
        default=20,
        type=int,
    )
    parser.add_argument(
        '--sound',
        help='include sound in the output video.',
        default=False,
        action='store_true'
    )

    return parser


def log_to_tensorboard(frame, log_dir):
    # Numpy array
    np_image_data = np.asarray(frame)
    # maybe insert float convertion here
    np_final = np.expand_dims(np_image_data, axis=0)

    # Add image summary
    summary_op = tf.summary.image("example", np_final)

    # Session
    with tf.Session() as sess:
        # Run
        summary = sess.run(summary_op)
        # Write summary
        writer = tf.summary.FileWriter(log_dir)
        writer.add_summary(summary)
        writer.close()


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not args.models_dir:
        raise RuntimeError(
            '--model-dir required. Please point '
            '--model-dir to directory where face recognition models are in.'
        )

    if not args.file:
        raise RuntimeError(
            '--file required. Please point '
            '--file to an input video file.'
        )

    logging.info('Start video processing: %s', datetime.datetime.now())

    face_recognition.set_face_recognition_models(args.models_dir)

    # Open videofile
    video = cv2.VideoCapture(args.file)
    fps = video.get(cv2.CAP_PROP_FPS)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Remove file if any
    try:
        os.remove(args.output)
    except:
        pass

    # Read codec information from input video.
    ex = int(video.get(cv2.CAP_PROP_FOURCC))
    codec = (
        chr(ex & 0xFF) +
        chr((ex & 0xFF00) >> 8) +
        chr((ex & 0xFF0000) >> 16) +
        chr((ex & 0xFF000000) >> 24)
    )

    cuda = built_with_cuda()
    if not cuda:
        codec = 'MP4V'

    logging.info('Create video %sx%s with FPS %s and CODEC=%s' % (width, height, fps, codec))
    fourcc = cv2.VideoWriter_fourcc(*codec)
    output_movie = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    frame_number = 0

    if args.log_images != 0:
        log_every_frame = int(length / args.log_images)
        logging.info('Will log every %s frame to tensorboard.' % log_every_frame)
    else:
        log_every_frame = 1000000000

    while True:
        ret, frame = video.read()

        # Quit when the input video file ends
        if not ret:
            break

        # Convert frame from BGR to RGB
        rgb_frame = frame[:, :, ::-1]
        frame_number += 1

        # Resize frame of video to 1/4 size for faster face detection processing
        # small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame, model=args.model)

        # Modify input frame
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            # top *= 1
            # right *= 1
            # bottom *= 1
            # left *= 1

            # Extract the region of the image that contains the face
            face_image = frame[top:bottom, left:right]

            # Blur the face image
            face_image = cv2.GaussianBlur(face_image, (61, 61), 30)

            # Put the blurred face region back into the frame image
            frame[top:bottom, left:right] = face_image

        # Write the resulting image to the output video file
        logging.info("Writing frame {} / {}".format(frame_number, length))

        if frame_number % log_every_frame == 0:
            # Convert frame into RGB again
            rgb_frame = frame[:, :, ::-1]
            logging.info("Log frame %s to tensorboard", frame_number)
            log_to_tensorboard(rgb_frame, args.train_dir)

        output_movie.write(frame)

    # All done!
    video.release()
    output_movie.release()
    cv2.destroyAllWindows()

    logging.info('End video processing: %s', datetime.datetime.now())

    if args.sound:
        time.sleep(0.2)
        logging.info('Start merge audio: %s', datetime.datetime.now())
        merge_audio_with(args.file, args.output)
        logging.info('End merge audio: %s', datetime.datetime.now())


def built_with_cuda():
    b = cv2.getBuildInformation()
    lines = b.split('\n')

    for l in lines:
        if ' NVIDIA CUDA' in l:
            return l.split(':')[-1].strip().startswith('YES')

    return False


def merge_audio_with(original_video_file, target_video_file):
    dirname = tempfile.gettempdir()
    audio_file = path.join(dirname, 'audio')

    # Get audio codec
    # cmd = (
    #     'ffprobe -show_streams -pretty %s 2>/dev/null | '
    #     'grep codec_type=audio -B 5 | grep codec_name | cut -d "=" -f 2'
    #     % original_video_file
    # )
    # codec_name = subprocess.check_output(["bash", "-c", cmd]).decode()
    # codec_name = codec_name.strip('\n ')
    # audio_file += ".%s" % codec_name
    audio_file += ".%s" % "mp3"

    # Something wrong with original audio codec; use mp3
    # -vn -acodec copy file.<codec-name>
    cmd = 'ffmpeg -y -i %s -vn %s' % (original_video_file, audio_file)
    code = subprocess.call(shlex.split(cmd))
    if code != 0:
        raise RuntimeError("Failed run %s: exit code %s" % (cmd, code))

    # Get video offset
    cmd = (
        'ffprobe -show_streams -pretty %s 2>/dev/null | '
        'grep codec_type=video -A 28 | grep start_time | cut -d "=" -f 2'
        % original_video_file
    )
    video_offset = subprocess.check_output(["bash", "-c", cmd]).decode()
    video_offset = video_offset.strip('\n ')

    # Get audio offset
    cmd = (
        'ffprobe -show_streams -pretty %s 2>/dev/null | '
        'grep codec_type=audio -A 28 | grep start_time | cut -d "=" -f 2'
        % original_video_file
    )
    audio_offset = subprocess.check_output(["bash", "-c", cmd]).decode()
    audio_offset = audio_offset.strip('\n ')

    dirname = tempfile.gettempdir()
    video_file = path.join(dirname, 'video')

    # Get video codec
    cmd = (
        'ffprobe -show_streams -pretty %s 2>/dev/null | '
        'grep codec_type=video -B 5 | grep codec_name | cut -d "=" -f 2'
        % original_video_file
    )
    codec_name = subprocess.check_output(["bash", "-c", cmd]).decode()
    codec_name = codec_name.strip('\n ')
    video_file += ".%s" % codec_name

    shutil.copyfile(target_video_file, video_file)
    # subprocess.call(["cp", target_video_file, video_file])
    time.sleep(0.2)

    cmd = (
        'ffmpeg -y -itsoffset %s -i %s '
        '-itsoffset %s -i %s -c copy %s' %
        (video_offset, video_file, audio_offset, audio_file, target_video_file)
    )

    code = subprocess.call(shlex.split(cmd))
    if code != 0:
        raise RuntimeError("Saving video with sound failed: exit code %s" % code)


if __name__ == '__main__':
    main()
