import argparse
import os

import cv2
import numpy as np
import face_recognition
import tensorflow as tf


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
        default=20,
        type=int,
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
        writer = tf.summary.FileWriter(log_dir, flush_secs=30)
        writer.add_summary(summary)
        writer.close()


def main():
    parser = get_parser()
    args = parser.parse_args()

    face_recognition.set_face_recognition_models(args.models_dir)

    # Open videofile
    video = cv2.VideoCapture(args.file)
    fps = video.get(cv2.CAP_PROP_FPS)
    length = video.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Remove file if any
    try:
        os.remove(args.output)
    except:
        pass

    print('Create video %sx%s with FPS %s' % (width, height, fps))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_movie = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    frame_number = 0
    log_every_frame = int(length / args.log_images)

    print('Will log every %s frame to tensorboard.' % log_every_frame)

    while True:
        ret, frame = video.read()

        # Convert frame from BGR to RGB
        rgb_frame = frame[:, :, ::-1]
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

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
            face_image = cv2.GaussianBlur(face_image, (33, 33), 30)

            # Put the blurred face region back into the frame image
            frame[top:bottom, left:right] = face_image

        # Write the resulting image to the output video file
        print("Writing frame {} / {}".format(frame_number, length))

        if frame_number % log_every_frame == 0:
            log_to_tensorboard(frame, args.train_dir)

        output_movie.write(frame)

    # All done!
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
