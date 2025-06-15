import argparse
import os
import json
import xml.etree.ElementTree as ET

def convert_ann_xml2json(textListFile, xmlAnnDir, jsonAnnFile):
    with open(textListFile, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    kaist_annotation = {
        "info": {
            "dataset": "KAIST Multispectral Pedestrian Benchmark",
            "url": "https://soonminhwang.github.io/rgbt-ped-detection/",
            "related_project_url": "http://multispectral.kaist.ac.kr",
            "publish": "CVPR 2015"
        },
        "info_improved": {
            "sanitized_annotation": {
                "publish": "BMVC 2018",
                "url": "https://li-chengyang.github.io/home/MSDS-RCNN/",
                "target": "files in train-all-02.txt (set00-set05)"
            },
            "improved_annotation": {
                "url": "https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn",
                "publish": "BMVC 2016",
                "target": "files in test-all-20.txt (set06-set11)"
            }
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "person"},
            {"id": 1, "name": "cyclist"},
            {"id": 2, "name": "people"},
            {"id": 3, "name": "person?"}
        ]
    }

    image_id = 0
    annotation_id = 0

    for line in lines:
        image_path = line.strip()
        image_name = image_path.split('/')[-1].replace('.jpg', '')
        annotation_file = os.path.join(xmlAnnDir, image_name + '.xml')

        # 이미지 정보 추가
        kaist_annotation['images'].append({
            "id": image_id,
            "im_name": image_name + '.jpg',
            "height": 512,
            "width": 640
        })

        if os.path.exists(annotation_file):
            try:
                tree = ET.parse(annotation_file)
                root = tree.getroot()

                for obj in root.findall('object'):
                    name = obj.find('name')
                    if name is None:
                        continue
                    bbox = obj.find('bndbox')
                    if bbox is None:
                        continue

                    # xmin, ymin, xmax, ymax 방식인지 x, y, w, h 방식인지 확인
                    xmin = bbox.find('xmin')
                    ymin = bbox.find('ymin')
                    xmax = bbox.find('xmax')
                    ymax = bbox.find('ymax')

                    if xmin is not None and ymin is not None and xmax is not None and ymax is not None:
                        xmin = float(xmin.text)
                        ymin = float(ymin.text)
                        xmax = float(xmax.text)
                        ymax = float(ymax.text)
                        width = xmax - xmin
                        height = ymax - ymin
                    else:
                        # x, y, w, h 방식
                        x = bbox.find('x')
                        y = bbox.find('y')
                        w = bbox.find('w')
                        h = bbox.find('h')
                        if x is None or y is None or w is None or h is None:
                            continue
                        xmin = float(x.text)
                        ymin = float(y.text)
                        width = float(w.text)
                        height = float(h.text)

                    # 카테고리 매핑
                    category_name = name.text
                    category_id = next((item['id'] for item in kaist_annotation['categories'] if item["name"] == category_name), None)
                    if category_id is None:
                        continue

                    # occlusion, ignore (optional)
                    occlusion = int(obj.find('occlusion').text) if obj.find('occlusion') is not None else 0
                    ignore = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0

                    # 어노테이션 추가
                    kaist_annotation['annotations'].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [xmin, ymin, width, height],
                        "height": height,
                        "occlusion": occlusion,
                        "ignore": ignore
                    })
                    annotation_id += 1

            except Exception as e:
                print(f"Error parsing {annotation_file}: {e}")

        image_id += 1

    with open(jsonAnnFile, 'w') as f:
        json.dump(kaist_annotation, f, indent=4)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--textListFile", type=str, default='datasets/kaist-rgbt/val.txt', help="Text file containing image file names (e.g., train-all-04.txt)")
    parser.add_argument("--xmlAnnDir", type=str, default='datasets/kaist-rgbt/train/labels-xml', help="XML annotation directory")
    parser.add_argument("--jsonAnnFile", type=str, default='datasets/kaist-rgbt/KAIST_coco23_annotation.json', help="Output json filename")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_opt()
    convert_ann_xml2json(args.textListFile, args.xmlAnnDir, args.jsonAnnFile)
