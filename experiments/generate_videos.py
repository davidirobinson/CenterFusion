
import os
from tqdm import tqdm
import argparse
import cv2 as cv
from natsort import os_sorted


fps_ = 15
ext_ = ".avi"
enc_ = "MJPG"

def get_image_size(video_name, input_dir):
    files = os.listdir(input_dir)
    for file in files:
        if video_name in file:
            img = cv.imread(input_dir + "/" + file)
            return (img.shape[1], img.shape[0])


def generate_video(video_name, input_dir, output_dir):

    size = get_image_size(video_name, input_dir)

    print("Generating", video_name, "of size", size)

    video = cv.VideoWriter(
        output_dir + "/" + video_name + ext_, cv.VideoWriter_fourcc(*enc_), fps_, size)

    files = os_sorted(os.listdir(input_dir))
    for i in tqdm(range(len(files))):
        file = files[i]
        if video_name in file:
            img = cv.imread(input_dir + "/" + file)
            video.write(img)

    video.release()


def main():
    """main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, required=False, help="path to input dir")
    parser.add_argument("-o", "--output-dir", type=str, required=False, help="path to output dir")
    args = parser.parse_args()

    generate_video("bird_pred", args.input_dir, args.output_dir)
    generate_video("ddd_pred", args.input_dir, args.output_dir)
    generate_video("generic", args.input_dir, args.output_dir)
    generate_video("pc_pillar_2d_blank", args.input_dir, args.output_dir)
    generate_video("pc_pillar_2d_inp", args.input_dir, args.output_dir)
    generate_video("pc_pillar_2d_ori", args.input_dir, args.output_dir)
    generate_video("pc_pillar_2d_out", args.input_dir, args.output_dir)
    generate_video("pc_pillar_2d_overlay", args.input_dir, args.output_dir)
    generate_video("pc_pillar_3d", args.input_dir, args.output_dir)
    generate_video("pred_hm", args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
