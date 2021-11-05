import json
import math
import numpy as np


def move_dog(json_name, json_dir):
    with open(f'{json_dir}/{json_name}.json', 'r', encoding='UTF-8-sig') as f:
        json_data = json.load(f)

    # pre_x = json_data["frames"][0]["frame"][0]["center_x"]
    # pre_y = json_data["frames"][0]["frame"][0]["center_y"]
    pre_x = 0
    pre_y = 0
    pre_no = 0

    standard = 35
    outlier = 60
    
    hl_dis = []

    for i in json_data["frames"] :

        if i["frame"] != []:
            for j in i["frame"]:
                
                move_x = abs(j["center_x"] - pre_x)
                move_y = abs(j["center_y"] - pre_y)
                
                compare_move = pow(move_x,2) + pow(move_y,2)
                # 속도측정 velocity
                if i["frame_no"] != pre_no:
                    velocity = (math.sqrt(compare_move))/(i["frame_no"]-pre_no)

                pre_x = j["center_x"]
                pre_y = j["center_y"]
                pre_no = i["frame_no"]

               
                

                if velocity > standard and velocity < outlier:
                    velocity = math.sqrt(compare_move)
                    # print("이동거리 x,y : ", move_x, move_y, "이동한거리 : ", velocity)
                    asdf = {
                        "이동거리" : velocity,
                        "frame_no" : i["frame_no"]
                    }
                    hl_dis.append(asdf)
       
        else:
            pass
    
    hl_dis = sorted(hl_dis, key=lambda x:x['이동거리'], reverse=True)
    HL_frame = []
    
    for i in hl_dis:
        a = i["frame_no"]
        
        if HL_frame == []:
            HL_frame.append(a)
        else:
            for j in HL_frame:
                if len(HL_frame) == 2: 
                    break
                elif abs(j-a) > 20: # 이곳에 flag(True나 False쓰는법)
                    HL_frame.append(a)
    print(HL_frame,type(HL_frame))
    print("하이라이트 기준 프레임넘버",HL_frame)

    return HL_frame




        
if __name__== '__main__':
    json_name = "dog_sample"
    move_dog(json_name)
