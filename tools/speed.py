# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from os.path import isfile
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config


def main(config, input_shape=(320, 320), batch_size=1, cfg_path="./config.yml", ordinal=1, nloop=100):
    devices = ['cpu', 'cuda:0']
    # devices = ['cpu']
    stored_speed_file = cfg_path + f'-b{batch_size}-log_speed.rec'
    log_path = cfg_path + f"-b{batch_size}-speed.log"
    if ordinal > 1 and isfile(stored_speed_file):
        log_file = open(log_path, "a")
        with open(stored_speed_file, 'rb') as handle:
            speeds = pickle.load(handle)
    else:
        log_file = open(log_path, "w")
        speeds = {}
        flops = {}
        params = {}
        for device in devices:
            speeds[device] = {}
            speeds[device]["batch_size"] = batch_size
            speeds[device]["elapsed"] = []
            speeds[device]["latency"] = []
            speeds[device]["Hz"] = []
        model_info = "Input size: (%s, %s)\n" % (input_shape[0], input_shape[1])
        log_file.write(model_info)
        log_file.write("Config: %s \n" % cfg_path)

    model = build_model(config.model)
    header = "\n" + "=" * 20 + "Ordinal: " + str(ordinal) + "=" * 20 + "\n"
    log_file.write(header)
    print(header)
    
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    
    img = np.random.rand(input_shape[0], input_shape[1], 3)
    img_info = {"id": 0}
    img_info["file_name"] = None
    img_info["id"] = 0

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    meta = dict(img_info=img_info, raw_img=img, img=img)
    meta = pipeline(None, meta, config.data.val.input_size)
    
    with torch.no_grad():
        for device in devices:
            meta = dict(img_info=img_info, raw_img=img, img=img)
            meta = pipeline(None, meta, config.data.val.input_size)
            meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(device)
            meta = naive_collate([meta for i in range(batch_size)])
            meta["img"] = stack_batch_img(meta["img"], divisible=32)
            model = model.to(device).eval()
            # warm up
            for i in range(5):
                results = model.inference(meta)
            t_start = time.time()
            for i in range(nloop):
                results = model.inference(meta)
            elapsed_time = time.time() - t_start
            speeds[device]["elapsed"].append(elapsed_time)
            speeds[device]["latency"].append(elapsed_time * 1000 / (nloop * batch_size))
            speeds[device]["Hz"].append(nloop * batch_size / elapsed_time)
  
        for device, speed in speeds.items():
            header = "*" * 5 + "Device: " + str(device) + "***Batch size: " + str(speeds[device]["batch_size"]) + "*" * 5
            print(header)
            print('\tElap: %.2f [s]' % (np.mean(speeds[device]["elapsed"])))
            print('\tLat: %.2f [ms]' % (np.mean(speeds[device]["latency"])))
            print('\tHz: %.2f [hz]\n' % (np.mean(speeds[device]["Hz"])))
            log_file.write(header)
            log_file.write('\n\tElap: %.2f [s]' % (np.mean(speeds[device]["elapsed"])))
            log_file.write('\n\tLat: %.2f [ms]' % (np.mean(speeds[device]["latency"])))
            log_file.write('\n\tHz: %.2f [hz]\n' % (np.mean(speeds[device]["Hz"])))
    log_file.write("\n")
    print("Speed log: ", speeds)
    with open(stored_speed_file, 'wb') as handle:
        pickle.dump(speeds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    log_file.close()
    


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth model to onnx.",
    )
    parser.add_argument("cfg", type=str, help="Path to .yml config file.")
    parser.add_argument(
        "--input_shape", type=str, default=None, help="Model intput shape."
    )
    parser.add_argument(
        "--device", type=int, default=-1, help="Device for test speed."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch for test speed."
    )
    parser.add_argument(
        "--nloop", type=int, default=100, help="Number of loop for test speed."
    )
    parser.add_argument(
        "--ordinal", type=int, default=1, help="Ordinal of running.."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg_path = args.cfg
    load_config(cfg, cfg_path)

    input_shape = args.input_shape
    if input_shape is None:
        input_shape = cfg.data.train.input_size
    else:
        input_shape = tuple(map(int, input_shape.split(",")))
        assert len(input_shape) == 2
    main(config=cfg, input_shape=input_shape, cfg_path=cfg_path, ordinal=args.ordinal, nloop=args.nloop)