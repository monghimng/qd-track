import argparse
import os
import os.path as osp
from collections import defaultdict

import mmcv
from tqdm import tqdm

CLASSES = [
    'pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle',
    'train'
]
USELESS = ['traffic light', 'traffic sign']
IGNORES = ['trailer', 'other person', 'other vehicle']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert BDD100K tracking label to COCO-VID format')
    parser.add_argument('-i', '--input', help='path of BDD label json file')
    parser.add_argument(
        '-o', '--output', help='path to save coco formatted label file')
    return parser.parse_args()


"""
Expected Tao directory structure to be:

tao_root
    frames
    annotations

It is recommended to create a tao_root/annotations_coco to house the reformatted data.
Generate a dataset in the following format:

categories: [
    {
        id: 1-based index,
        name: str describing the category
    }
]

videos: [
    {
        id: 0-based index,
        name: yyy
    },
]
    
images: [
    {
        file_name: relative path of the img,
        height:
        width:
        id; 0-based id unique to each individual img,
        video_id: correspond id from the videos json,
        frame_id: todo
    }
]

annotations: [
    {
        id: 0-based global unique identifier,
        image_id: corresponding image id it is in,
        category_id:
        instance_id: id to track object,
        bbox:
        area:
        occluded:
        truncated:
        iscrowd:
    }
]
"""
def main():
    """
    All we need to change are
        1. tao frame_index refers to indices of the frame with some missing frames in-bet, we want
            simply its 0-based index within the video
        2. rename track_id to instance_id for each annotation
    """
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    for subset in ['train', 'validation']:

        lvis_classes_anno_only = f'{subset}.json'
        all_classes_anno = f'{subset}_with_freeform.json'

        for info in [lvis_classes_anno_only, all_classes_anno]:

            path = osp.join(args.input, info)
            print(f'convert tao tracking {path} set into COCO-VID format')
            coco = mmcv.load(path)

            # import pdb;pdb.set_trace()
            # [a['id'] for a in coco['images']][:100]
            # [a['frame_index'] for a in coco['images']][:100]
            # [a['frame_index'] for a in vid_id2img_infos[0]][:100]

            vid_id2img_infos = defaultdict(list)
            for i, img in enumerate(coco['images']):
                vid_id2img_infos[img['video_id']].append(img)

            for img_infos_within_a_vid in vid_id2img_infos.values():
                img_infos_within_a_vid = sorted(img_infos_within_a_vid, key=lambda img_info: img_info['frame_index'])
                for i, img_info in enumerate(img_infos_within_a_vid):
                    img_info['frame_id'] = i
                    img_info['frame_index'] = i

            for anno in coco['annotations']:
                # for some reason, this codebase expects instance_id to be 1-based
                anno['instance_id'] = anno['track_id'] + 1
                anno['track'] = anno['track_id'] + 1

            mmcv.dump(
                coco,
                osp.join(args.output, info))

            ############################## generate a small version for debugging, containing only the first video

            coco['videos'] = [[vid for vid in coco['videos'] if vid['metadata']['dataset'] == 'BDD'][0]]
            vid = coco['videos'][0]['id']
            coco['images'] = [img for img in coco['images'] if img['video_id'] == vid]
            img_ids = {img['id'] for img in coco['images']}
            coco['annotations'] = [ann for ann in coco['annotations'] if ann['image_id'] in img_ids]
            mmcv.dump(
                coco,
                osp.join(args.output, 'small_' + info))

if __name__ == '__main__':
    main()

"""
deactivate
source $PYTHON_ENV/conda/bin/activate
conda activate qdtrack
python $CODE/qd-track/tools/convert_datasets/taotrack2cocovid.py -i /data/ck/data/tao/annotations -o /data/ck/data/tao/annotations_coco


############################## code scraps for examining the integrity of tao
[a['name'] for a in coco['categories']]
[a['name'] for a in coco['categories']][206]  # is cap_(headwear)
coco['categories'][979]  # skateboard

cid = [a['category_id'] for a in coco['annotations']]
import numpy as np
cid = np.array(cid)
ckkk = [print(i) for i, c in enumerate(cid) if c == 804]
ckkk = [print(i) for i, c in enumerate(cid) if c == 805]
ckkk = [print(i) for i, c in enumerate(cid) if c == 979]

ckkk = [print(i) for i, c in enumerate(cid) if c == 980]

coco['annotations'][50238]
coco['annotations'][50243]
coco['annotations'][50248]
coco['annotations'][50257]

coco['annotations'][54611]
coco['annotations'][54141]

coco['annotations'][47224]

[img for img in coco['images'] if img['id'] == 98189]
[img for img in coco['images'] if img['id'] == 93355]
##############################
"""


