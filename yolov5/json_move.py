import json
import math
import numpy as np


def move_dog(json_name, save_dir):
    with open(f'{save_dir}/{json_name}.json', 'r', encoding='UTF-8-sig') as f:
        json_data = json.load(f)

    # pre_x = json_data["frames"][0]["frame"][0]["center_x"]
    # pre_y = json_data["frames"][0]["frame"][0]["center_y"]
    pre_x = 0
    pre_y = 0
    pre_no = 0

    standard = 2
    outlier = 20
    
    hl_dis = []

    for i in json_data["frames"] :

        if i["frame"] != []:
            for j in i["frame"]:
                
                move_x = abs(j["center_x"] - pre_x)
                move_y = abs(j["center_y"] - pre_y)
                
                compare_move = pow(move_x,2) + pow(move_y,2)
                # 몇프레임동안 얼마의거리 움직였는지 = distance
                if i["frame_no"] != pre_no:
                    distance = (math.sqrt(compare_move))/(i["frame_no"]-pre_no)

                pre_x = j["center_x"]
                pre_y = j["center_y"]
                pre_no = i["frame_no"]

               
                

                if distance > standard and distance < outlier:
                    distance = math.sqrt(compare_move)
                    # print("이동거리 x,y : ", move_x, move_y, "이동한거리 : ", distance)
                    asdf = {
                        "이동거리" : distance,
                        "frame_no" : i["frame_no"]
                    }
                    hl_dis.append(asdf)
       
        else:
            pass
    
    hl_dis = sorted(hl_dis, key=lambda x:x['이동거리'], reverse=True)
    pass_dis = []
    
    for i in hl_dis:
        a = i["frame_no"]
        
        if pass_dis == []:
            pass_dis.append(a)
        else:
            for j in pass_dis:
                if len(pass_dis) == 2: 
                    break
                elif abs(j-a) > 1000: # 이곳에 flag(True나 False쓰는법)
                    pass_dis.append(a)
    print(pass_dis,type(pass_dis))
    print(hl_dis)

    return pass_dis




        
if __name__== '__main__':
    json_name = "dog_sample"
    move_dog(json_name)