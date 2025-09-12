import pandas as pd
import json
import re

# with open('/{dir_path}/datasets/conceptual_caption/cc3m.json', 'r') as file:
#     my_list = json.load(file)
# print(my_list[0])
# exit()

with open('/{dir_path}/datasets/conceptual_caption/cc595k_raw.json', 'r') as file:
    my_list = json.load(file)
print(my_list[0])

res = []
cnt = 0
for dic in my_list:
    image, caption, blip_caption = dic['image'], dic['caption'], dic['blip_caption']
    if blip_caption is None:
        print(image, "****", caption, "****", blip_caption)
        cnt += 1
        blip_caption = caption
    res.append({
        'image': "/{dir_path}/datasets/conceptual_caption/cc3m_595k_images/"+image,
        'caption': blip_caption,
    })
    
    
with open('/{dir_path}/datasets/conceptual_caption/cc595k.json', 'w') as json_file:
    json.dump(res, json_file)
print("finished!", len(res), "but", cnt, "blip_caption is none")
    