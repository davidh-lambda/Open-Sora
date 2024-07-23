import re
import os
import glob
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import wandb
from queue import Queue, Empty
import time
import argparse

wandb.require("core")

num_gpus = 1
project = "que_sora_sora12"
save_to_new = False
compute_only = True

api = wandb.Api()
runs = list(api.runs())
expnum_pattern = re.compile(r'(\d+)-')
step_pattern = re.compile(r'.*global_step(\d+)')
def get_run_id(expnum):
    for run in runs:
        if "(" in run.name and ")" in run.name:
            if "(%03i)" % expnum in run.name:
                return run.id
        else:
            if run.name.startswith("%03i-" % expnum):
                return run.id

# Lock to ensure each GPU is used by only one thread at a time
gpu_locks = [threading.Lock() for _ in range(num_gpus)]

def normalize_path_name(v):
    return v.replace(":", "-")
def create_output_directory(input_name, settings):
    settingsstr = f"{'_'.join(f'{normalize_path_name(v)}' for k, v in settings.items())}"
    output_dir = input_name.replace("outputs/", f"samples/{settingsstr}/")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, settingsstr

def run_script(settings, global_settings, input_name, prompts, gpu_id):
    output_dir, settingsstr = create_output_directory(input_name, settings)

    # skip creation if already done
    output_file = os.path.join(output_dir, "sample_%04i.mp4" % (len(prompts) - 1))
    if not save_to_new and os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping...")
        return output_file
    
    # limit worker to gpu_id
    env = os.environ.copy()
    #env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # actually do the inference
    if not os.path.exists(output_file):
        full_settings = {**settings, **global_settings, "ckpt-path": input_name, "save-dir": output_dir}
        script_args = [f"--{k} {v}" for k, v in full_settings.items()]
        script_args = " ".join(script_args)
        command = f"python scripts/inference.py configs/opensora-v1-2/inference/sample.py {script_args}"
        print(f"Running: {command} on GPU {gpu_id}")
        result = subprocess.run(command, shell=True, env=env)
        if result.returncode != 0:
            print(f"Script failed with return code {result.returncode}")
            return None
    else:
        print("Skipping ", output_file)

    # extract expnum
    match = expnum_pattern.match(input_name.split("/")[-2])
    if not match:
        raise ValueError("Did not find expnum from input_dir %s" % input_name)
    expnum = int(match[1])

    # extract step
    match = step_pattern.match(input_name.split("/")[-1])
    if not match:
        raise ValueError("Did not find expnum from input_dir %s" % input_name)
    global_step = int(match[1])

    if compute_only:# or global_step < 400:
        return None

    # Upload to wandb
    run_id = get_run_id(expnum)
    if save_to_new:
        run = wandb.init(project=project, dir="./outputs/wandb")
    else:
        run = wandb.init(project=project, id=run_id, resume=True, dir="./outputs/wandb")
    run.log({
        #**{f"eval{'' if settingstr == '4s_360p_9-16' else '_' + settingsstr }/{prompt_i}": wandb.Video(
        **{f"eval/{prompt_i}": wandb.Video(
            #**{f"eval-hd/prompt_{prompt_i}": wandb.Video(
            os.path.join(output_dir, "sample_%04i.mp4" % prompt_i),
            caption=prompt,
            format="mp4",
            fps=24,
            ) for prompt_i, prompt in enumerate(prompts)},
        #"eval{'' if settingstr == '4s_360p_9-16' else '/' + settingsstr }_step": global_step,
        },
        #**({} if not save_to_new else {"step": global_step}),
        #**({"step": global_step}),
        commit=True
    )
    
    return output_dir

def worker(gpu_id, job_queue):
    while not job_queue.empty():
        try:
            args = job_queue.get_nowait()
        except Empty:
            break
        with gpu_locks[gpu_id]:
            run_script(*args, gpu_id)
        job_queue.task_done()

def main():
    settings_presets = {
        "low4s": [{"num-frames": "4s", "resolution": "360p", "aspect-ratio": "9:16"}],
        "low16s": [{"num-frames": "16s", "resolution": "360p", "aspect-ratio": "9:16"}],
        "high4s": [{"num-frames": "4s", "resolution": "720p", "aspect-ratio": "9:16"}]
    }

    parser = argparse.ArgumentParser(description="Process some settings and expnums.")
    parser.add_argument('--expnums', nargs='+', type=int, required=True, help='List of experiment numbers')
    parser.add_argument('--preset', choices=settings_presets.keys(), required=True, help='Preset settings to use')
    args = parser.parse_args()

    settings_list = settings_presets[args.preset]


    global_settings = {
        "prompt-path": "./samples/prompts.txt",
        "batch-size": 4
        # "prompt-as-path": True
    }

    #expnums = [17,19,20,21,22,34,43,46,73,74]
    #expnums = [88]
    #expnums = [163, 164, 165, 168, 174, 175, 176, 177]
    expnums = args.expnums

    with open(global_settings["prompt-path"], 'r') as fp:
        prompts = [p.strip() for p in fp.readlines()]

    while True:
        print("checking ...")

        input_names = []
        for expnum in expnums:
            print("checking", "./outputs/%03d-STDiT3-XL-2/epoch*" % expnum)
            input_names.extend(glob.glob("./outputs/%03d-STDiT3-XL-2/epoch*" % expnum))

        # DEBUG
        #run_script(settings_list[0], global_settings, input_names[0], prompts, 0)
        #return

        # Create a queue with all the jobs
        job_queue = Queue()
        for input_name in input_names:
            for settings in settings_list:
                print("tasks created for:", input_name, settings)
                print(settings)
                job_queue.put((settings, {**global_settings, **({} if settings["num-frames"] == "4s" else {"batch-size": 2})}, input_name, prompts))
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            for gpu_id in range(num_gpus):
                executor.submit(worker, gpu_id, job_queue)
            
            job_queue.join()

        time.sleep(60)

    print("All tasks completed")

if __name__ == "__main__":
    main()
