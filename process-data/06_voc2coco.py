import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re

def get_map_dict():
    new_class_dict = {}
    new_class_dict['rice'] = ['rice',]
    new_class_dict['rice bowl'] = ['eels on rice',"chicken-'n'-egg on rice", "pork cutlet on rice", "tempura bowl","bibimbap","beef bowl"]
    new_class_dict['fried rice'] = ['pilaf','chicken rice','fried rice',"mixed rice"]
    new_class_dict['curry bowl'] = ['beef curry',"cutlet curry"]
    new_class_dict['sushi'] = ["sushi"]
    new_class_dict['toast'] = ["toast","pizza toast"]
    new_class_dict['croissant'] = ["croissant"]
    new_class_dict['bread'] = ["roll bread","raisin bread",]
    new_class_dict['hamburger'] = ["hamburger",]
    new_class_dict['pizza'] = ["pizza",]
    new_class_dict['sandwiches'] = ["sandwiches",]
    new_class_dict['noodle or ramen'] = ["udon noodle","tempura udon","soba noodle","ramen noodle","beef noodle","tensin noodle","dipping noodles"]
    new_class_dict['fried noodle'] = ["fried noodle"]
    new_class_dict['spaghetti'] = ["spaghetti","spaghetti meat sauce"]
    new_class_dict['stir fry or boiled vegetables'] = ["sauteed vegetables","grilled eggplant","sauteed spinach","kinpira-style sauteed burdock","goya chanpuru"]
    new_class_dict['omelet'] = ["omelet","rolled omelet","omelet with fried rice"]
    new_class_dict['dumpling'] = ["jiaozi","steamed meat dumpling"]
    new_class_dict['stew'] = ["stew","seasoned beef with potatoes"]
    new_class_dict['fish'] = ["teriyaki grilled fish","grilled salmon","salmon meuniere","grilled pacific saury","lightly roasted fish","nanbanzuke","boiled fish","dried fish","fried fish",]
    new_class_dict['sashimi'] = ["sashimi",]
    new_class_dict['hot pot'] = ["sukiyaki",]
    new_class_dict['stir fried or boil meat'] = ["sweet and sour pork","ginger pork saute","stir-fried beef and peppers","boiled chicken and vegetables",]
    new_class_dict['fried chicken'] = ["fried chicken",]
    new_class_dict['steak'] = ["hambarg steak","beef steak"]
    new_class_dict['egg'] = ["egg sunny-side up",]
    new_class_dict['shrimp'] = ["shrimp with chill source","fried shrimp"]
    new_class_dict['roast chicken'] = ["roast chicken",]
    new_class_dict['salad'] = ["potato salad","green salad","macaroni salad"]
    new_class_dict['soup'] = ["Japanese tofu and vegetable chowder","pork miso soup","chinese soup","miso soup"]
    new_class_dict['hot dog'] = ["hot dog"]
    new_class_dict['french fries'] = ["french fries"]
    new_class_dict['tofu'] = ["spicy chili-flavored tofu","cold tofu"]

    new_class_dict['chip butty'] = ["chip butty",]
    new_class_dict['Japanese-style pancake'] = ["Japanese-style pancake",]
    new_class_dict['takoyaki'] = ["takoyaki",]
    new_class_dict['gratin'] = ["gratin",]
    new_class_dict['croquette'] = ["croquette",]
    new_class_dict['tempura'] = ["tempura","vegetable tempura",]
    new_class_dict['potage'] = ["potage",]
    new_class_dict['sausage'] = ["sausage",]
    new_class_dict['oden'] = ["oden",]
    new_class_dict['ganmodoki'] = ["ganmodoki",]
    new_class_dict['steamed egg hotchpotch'] = ["steamed egg hotchpotch",]
    new_class_dict['sirloin cutlet'] = ["sirloin cutlet",]
    new_class_dict['skewer'] = ["yakitori",]
    new_class_dict['cabbage roll'] = ["cabbage roll",]
    new_class_dict['fermented soybeans'] = ["fermented soybeans",]
    new_class_dict['egg roll'] = ["egg roll",]
    new_class_dict['chilled noodle'] = ["chilled noodle",]
    new_class_dict['simmered meat'] = ["simmered pork",]
    new_class_dict['fish bowl'] = ["sushi bowl","sashimi bowl",]
    new_class_dict['fish-shaped pancake with bean jam'] = ["fish-shaped pancake with bean jam",]
    new_class_dict['rice ball'] = ["rice ball",]
    
    map_dict = {}
    for new_class in new_class_dict.keys():
        old_classes = new_class_dict[new_class]
        for old_class in old_classes:
            map_dict[old_class] = new_class
    
    return map_dict, new_class_dict

def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str  = []
        for i, line in enumerate(f):
            line = line.rstrip('\n')  # delete \n in the end of th
            # e line
            line = line.split('\t')
            labels_str.append(line[0])
            
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid+ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    if label[-1] == ' ':
        label = label[:-1]
        
    map_dict, new_class_dict = get_map_dict()
    try:
        new_label = map_dict[label]
    except:
        import pdb; pdb.set_trace()
    category_id = label2id[new_label]
    bndbox = obj.find('bndbox')
    
    try:
        xmin = int(float(bndbox.findtext('xmin'))) 
        ymin = int(float(bndbox.findtext('ymin'))) 
        xmax = int(float(bndbox.findtext('xmax')))
        ymax = int(float(bndbox.findtext('ymax')))
    except Exception as E:
        print(E)
        import pdb; pdb.set_trace()
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        try:
            ann_tree = ET.parse(a_path)
        except:
            continue
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def main():
    parser = argparse.ArgumentParser(
        description='This script support converting voc format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotation files directory. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_ids', type=str, default=None,
                        help='path to annotation files ids list. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_paths_list', type=str, default=None,
                        help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
    parser.add_argument('--labels', type=str, default=None,
                        help='path to label list.')
    parser.add_argument('--output', type=str, default='output.json', help='path to output json file')
    parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
    parser.add_argument('--extract_num_from_imgid', action="store_true",
                        help='Extract image number from the image filename')
    args = parser.parse_args()
    label2id = get_label2id(labels_path=args.labels)
    ann_paths = get_annpaths(
        ann_dir_path=args.ann_dir,
        ann_ids_path=args.ann_ids,
        ext=args.ext,
        annpaths_list_path=args.ann_paths_list
    )
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=args.output,
        extract_num_from_imgid=args.extract_num_from_imgid
    )


if __name__ == '__main__':
    main()
