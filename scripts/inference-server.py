import re
import os
import glob
import subprocess
import time
import argparse

import wandb

wandb.require("core")

project = "sora_speedrun"
ckpt_dir = "outputs_speedrun"

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




def run_script(settings, global_settings, input_name, prompts, last_step, compute_only, commit):
    settingsstr = '_'.join(f'{v.replace(":", "-")}' for k, v in settings.items())
    output_dir = input_name.replace(ckpt_dir + "/", f"samples/{settingsstr}/")
    os.makedirs(output_dir, exist_ok=True)

    # skip creation if already done
    output_file = os.path.join(output_dir, "sample_%04i.mp4" % (len(prompts) - 1))

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
    if last_step is not None and global_step <= last_step:
        return None
    
    if not os.path.exists(output_file):
        if expnum == 999:
            input_name += "/model.safetensors"
        full_settings = {**settings, **global_settings, "ckpt-path": input_name, "save-dir": output_dir}
        script_args = [f"--{k} {v}" for k, v in full_settings.items()]
        script_args = " ".join(script_args)
        command = f"python scripts/inference.py configs/opensora-v1-2/inference/sample.py {script_args}"
        result = subprocess.run(command, shell=True, env=os.environ.copy())
        if result.returncode != 0:
            return None
    else:
        if compute_only:  # or global_step < 400:
            return None

    # Upload to wandb
    eval_str = 'eval' + ('' if not settingsstr or settingsstr == '4s_360p_9-16' else '_' + settingsstr)
    print(f"sending logs to {eval_str}/0-{len(prompts)-1}")
    run.log({
        **{f"{eval_str}/{prompt_i}": wandb.Video(
        #**{f"eval/{prompt_i}": wandb.Video(
            #**{f"eval-hd/prompt_{prompt_i}": wandb.Video(
            os.path.join(output_dir, "sample_%04i.mp4" % prompt_i),
            caption=prompt,
            format="mp4",
            fps=24,
            ) for prompt_i, prompt in enumerate(prompts)},
        f"{eval_str}_step": global_step,
        },
        **({"step": global_step}),
        commit=commit
    )
    print("sending finished")
    
    return global_step



def main():
    settings_presets = {
        "low4s": [{"num-frames": "4s", "resolution": "360p", "aspect-ratio": "9:16"}],
        "low16s": [{"num-frames": "16s", "resolution": "360p", "aspect-ratio": "9:16"}],
        "high4s": [{"num-frames": "4s", "resolution": "720p", "aspect-ratio": "9:16"}]
    }

    parser = argparse.ArgumentParser(description="Process some settings and expnums.")
    parser.add_argument('--expnums', nargs='+', type=str, required=True, help='List of experiment numbers')
    parser.add_argument('--preset', choices=settings_presets.keys(), required=True, help='Preset settings to use')
    parser.add_argument('--compute_only', action='store_true', help='Only compute without uploading to wandb')
    parser.add_argument('--last_step', type=int, default=None, help='Last processed step')
    parser.add_argument('--save_to', default=None, help='Attach to run with id "save_to".')
    parser.add_argument('--only-first-of-epoch', action='store_true', help='Use only the first folder for each epoch')
    args = parser.parse_args()

    save_to_new = args.save_to == "new"
    compute_only = args.compute_only

    settings_list = settings_presets[args.preset]



    global_settings = {
        "prompt-path": "./samples/prompts.txt",
        #"batch-size": 4
        "batch-size": 8
        # "prompt-as-path": True
    }

    #expnums = [17,19,20,21,22,34,43,46,73,74]
    #expnums = [88]
    #expnums = [163, 164, 165, 168, 174, 175, 176, 177]
    if len(args.expnums) == 1 and os.path.isfile(args.expnums[0]):
        with open(args.expnums[0], 'r') as f:
            expnums = [int(line.strip()) for line in f if line.strip().isdigit()]
    else:
        expnums = [int(e) for e in args.expnums]

    with open(global_settings["prompt-path"], 'r') as fp:
        prompts = [p.strip() for p in fp.readlines()]

    last_step = args.last_step

    while True:

        # DEBUG
        #run_script(settings_list[0], global_settings, input_names[0], prompts, 0)
        #return

        global run

        weight_folders = []


        input_names = []
        for expnum in expnums:
            input_names += glob.glob("./" + ckpt_dir + "/%03d-STDiT3-XL-2/epoch*" % expnum)
        #input_names = sorted(input_names, key=lambda name: int(re.search(r'epoch(\d+)-', name).group(1)))

        filtered_input_names = []
        if args.only_first_of_epoch:
            epoch_dict = {}
            for name in input_names:
                epoch_index = int(re.search(r'epoch(\d+)-', name).group(1))
                if epoch_index not in epoch_dict:
                    epoch_dict[epoch_index] = name
            filtered_input_names = []
            for epoch_index in sorted(epoch_dict.keys()):
                filtered_input_names.append(epoch_dict[epoch_index])
            filtered_input_names.append(input_names[-1])
            print(filtered_input_names)
        else:
            filtered_input_names = input_names

        # init wandb
        if compute_only:
            run = None
        elif save_to_new:
            run = wandb.init(project=project, dir="./" + ckpt_dir + "/wandb")
        elif args.save_to:
            run = wandb.init(project=project, id=args.save_to, resume=True, settings=wandb.Settings(_disable_stats=True, _disable_meta=True))
        else:
            assert len(expnums) == 1
            run_id = get_run_id(expnum)
            run = wandb.init(project=project, dir="./" + ckpt_dir + "/wandb", id=run_id, resume=True, settings=wandb.Settings(_disable_stats=True, _disable_meta=True))

        if not save_to_new:
            filtered_input_names = reversed(filtered_input_names)
        for input_name in filtered_input_names:
            for settings in settings_list:
                job = (settings, {**global_settings, **({} if settings["num-frames"] == "4s" and settings["resolution"] == "360p" else {"batch-size": 2})}, input_name, prompts)
                is_last_setting = settings == settings_list[-1]
                last_step_new = run_script(*job, last_step, compute_only, is_last_setting)
                if last_step_new:
                    last_step = last_step_new

        time.sleep(60)


if __name__ == "__main__":
    main()
