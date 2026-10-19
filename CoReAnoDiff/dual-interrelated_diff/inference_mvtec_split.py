import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,StableDiffusionPipeline_bg,DPMSolverMultistepScheduler
# import ipdb
import sys
import os
from PIL import Image
sys.path.append('/path/to/DualAnoDiff/dual-interrelated_diff')
pipe = DiffusionPipeline.from_pretrained(
    "/path/to/stable-diffusion-v1-5", safety_checker=None
).to("cuda")
# #############
args = sys.argv
mvtec_name = args[1]
mvtec_aomaly_name = args[2]
# ##############
pipe.load_lora_weights('./all_generate/'+mvtec_name+'/'+mvtec_aomaly_name+'/checkpoint-5000')

target_path = './generate_data/'+mvtec_name+'/'+ mvtec_aomaly_name
if not os.path.exists(os.path.join(target_path,'image')):
    os.makedirs(os.path.join(target_path,'image'))
if not os.path.exists(os.path.join(target_path,'fg')):
    os.mkdir(os.path.join(target_path,'fg'))
cnt = len(os.listdir(os.path.join(target_path,'image')))

# for i in range(cnt,1000):
for i in range(cnt,500):
    images = pipe(prompt_blend='a vfx with sks',prompt_fg="sks",num_inference_steps=100,guidance_scale=2.5).images

    images[0].save(os.path.join(target_path,'image',str(i)+".png"))
    images[1].save(os.path.join(target_path,'fg',str(i)+".png"))
    print(i)
 
