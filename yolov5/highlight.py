import numpy as np
import cv2
import json
import argparse
import os
from pathlib import Path

from json_move import move_dog
from utils.general import increment_path


import torch

# def section():
#     with open("log/dog_vidio-1.json", "r") as f:
#         json_data = json.load(f)
#     print(json.dumps(json_data))





@torch.no_grad()
def Highlight(path,
            result_name,
            json_name,
            save_dir
            ):
    exist_ok = False
    save_txt = False
    Highlight = "runs/highlight"
    name = "highlight"
    
    save_dir_highlight = increment_path(Path(Highlight) / name, exist_ok=exist_ok)  # increment run
    (save_dir_highlight / 'labels' if save_txt else save_dir_highlight).mkdir(parents=True, exist_ok=True)  # make dir


    hl_dis = move_dog(json_name,save_dir)
    print(hl_dis)

    for i in range(len(hl_dis)):
        try:
            # mp4파일일경우
            codec=cv2.VideoWriter_fourcc(*'MP4V')
        except:
            # avi파일일 경우
            codec=cv2.VideoWriter_fourcc(*'FMP4')

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        size = (int(width), int(height))

        # vid_writer = cv2.VideoWriter(result_path, codec, fps, size)
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print('총 Frame 갯수:', frame_cnt, 'FPS:', round(fps), 'Frame 크기:', size)
            
        result_name = os.path.splitext(result_name)[0]
        result_path = f"{save_dir_highlight}/{result_name}_{i}.mp4"
        vid_writer = cv2.VideoWriter(result_path, codec, fps, size)


        while(cap.isOpened()):
            ret, image = cap.read()
            # cv2.imshow(path, image) # 동영상 확인용
            # cv2.waitKey(delay)
            
            if cap.get(1) > hl_dis[i] and cap.get(1) < hl_dis[i]+100:
                
    
                vid_writer.write(image)
            
            if not ret: # 동영상 끝나면 닫겠다.
                print(cap.get(1), type(cap.get(1)), result_path)
                break


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path', type=str, default='data/videos/dog_sample.mp4', help='load results to project/name')
#     parser.add_argument('--result_name', default='frame.mp4', help='save results to project/name')
#     opt = parser.parse_args()
#     print(opt)
#     return opt

# def main(opt):
#     Highlight(**vars(opt))


# if __name__ == '__main__':
#     opt = parse_opt()
#     main(opt)

    