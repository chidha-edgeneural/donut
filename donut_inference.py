from ast import parse
from donut import DonutModel
from PIL import Image
import argparse
import json
from time import time as timer
import os
import gradio as gr


pretrained_model = DonutModel.from_pretrained(
    os.environ["DONUT_CHECKPOINT"]    
)

TASK_NAME = os.environ["TASK_NAME"]

pretrained_model.half()
pretrained_model.to("cuda")

def get_results(img):
    output = pretrained_model.inference(
        Image.fromarray(img), #to load numpy array use Image.fromarray(np_arrayz)
        prompt=f"<s_{TASK_NAME}>"
    )
    
    result = output["predictions"][0]
    
    assesment_codes = []
    if "assesments" in list(result.keys()):
        for ele in result["assesments"]:
            ele.replace(",", " ")
            code = [x for x in ele.split("-")[1].split(" ") if x!=""][0]
            assesment_codes.append(code)
    result["extracted_codes"] = assesment_codes
    return result

demo = gr.Interface(
    fn=get_results,
    inputs=[
        gr.Image(label="Select Image"),
    ],
    outputs=[
        gr.JSON(label="Prediction Json Area"),
    ]
)

demo.launch(share=True)

print(f"Donut Load latency {m_load_latency}, Donut Inference Latency {m_infer_latency}")