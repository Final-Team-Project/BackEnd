import glob

input_dir = 'C:\DogDetect\yolov5\dog\labels'

f = open('./train_list.txt', 'w', encoding='utf-8')
input_file = glob.glob(input_dir + '*.txt')

#print(input_file)
for file in input_file:
    f.write(file[:41] + 'images/' + file[48:-4] + '.jpg \n')
f.close()