from ast import parse
from donut import DonutModel
from PIL import Image
import argparse
import json
from time import time as timer
import os

parser = argparse.ArgumentParser("Argument Handlers")
parser.add_argument("--img", type=str, required=True)
parser.add_argument("--output_json", type=str, required=True)
args = parser.parse_args()

image_path = args.img#"/home/ec2-user/chidha/doc-understanding/donut/doc_understanding_datasets/donut_dataset_rvl_cdip_modded/test/hsbc___6752158_HSBC Bank USA (21)_page-0007.jpg"

m_load_s = timer()
pretrained_model = DonutModel.from_pretrained(
    os.environ["DONUT_CHECKPOINT"]    
)
m_load_latency = timer() - m_load_s

TASK_NAME = os.environ["TASK_NAME"]

pretrained_model.half()
pretrained_model.to("cuda")

m_infer_s = timer()
output = pretrained_model.inference(
    Image.open(image_path), #to load numpy array use Image.fromarray(np_arrayz)
    prompt=f"<s_{TASK_NAME}>"
)

output = pretrained_model.inference(
    
)

m_infer_latency = timer() - m_infer_s

open(args.output_json, 'w').write(json.dumps(output))

print(f"Donut Load latency {m_load_latency}, Donut Inference Latency {m_infer_latency}")