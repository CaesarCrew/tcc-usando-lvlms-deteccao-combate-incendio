# Copyright (2023) Bytedance Ltd. and/or its affiliates
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

import argparse
from ruamel.yaml import YAML
import numpy as np
import random
import time
import datetime as dt
import json
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn

from utils import write_jsonl, write_txt_doc, get_rank
from k_folds import make_cross_validation
from calculate_metrics import process_metrics
from dataset import create_dataset, create_loader

torch.set_default_dtype(torch.float16)
@ torch.no_grad()
def evaluation(model, data_loader, device, config):
    model.eval() # test

    result = []

    for n, (idx, vision_input, input_ids, input_atts) in enumerate(data_loader):
        vision_input = vision_input.to(device, non_blocking=True)
        input_ids = input_ids[:, -128:]
        #input_ids = input_ids.to(device)
        input_atts = input_atts[:, -128:]
        input_atts = input_atts.to(device)#.half()

        with torch.amp.autocast('cuda'):
            text_outputs = model.generate(
                vision_input=vision_input,
                input_ids=input_ids, input_atts=input_atts,
                use_nucleus_sampling=config.get('use_nucleus_sampling', False),
                apply_lemmatizer=config['apply_lemmatizer'],
                num_beams=config['num_beams'],
                min_length=config['min_length'],
                length_penalty=config.get('length_penalty', 1.0),
                no_repeat_ngram_size=config.get('no_repeat_ngram_size', -1),
                top_p=config.get('top_p', 0.9),
                top_k=config.get('top_k', 3),
                max_new_tokens=config.get('max_new_tokens', 64))

        for i, output in zip(idx, text_outputs):
            result.append({"index": i, "text_output": output.strip()})
        torch.cuda.empty_cache()

    return result


def main(args, config):
    print("### Evaluating", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #print("config:", json.dumps(config), flush=True)

    print("### Creating model", flush=True)
    from models.lynx import LynxBase
    model = LynxBase(config=config, freeze_vit=config['freeze_vit'], freeze_llm=config['freeze_llm'], load_bridge=False)
    model.vision_encoder = model.vision_encoder.to(device).half()
    model.bridge = model.bridge.to(device).half()
    model.LLM = model.LLM.to("cpu")#.float()

    for _, param in model.named_parameters():
        param.requires_grad = False

    model.eval()

    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    print("### Creating datasets", flush=True)
    test_dataset = create_dataset('eval', config)

    #prediction_test_data = ''
    #total_time_str = ''

    if args.cross_val:
        k = args.k_folds
        assert k >= 2, "k_folds must be >= 2"
        predictions, total_time_str, per_fold_times, prediction_test_data = make_cross_validation(test_dataset, model, device, config, k, args.fold_seed, args.save_per_fold)
    else:
        start_time = time.time()
        print("### Start evaluating", flush=True)
        test_loader = create_loader([test_dataset], batch_size=[config['batch_size_test']], num_workers=[4],collate_fns=[test_dataset.collate_fn])[0]
        predictions = evaluation(model, test_loader, device, config)

        total_time = time.time() - start_time
        total_time_str = str(dt.timedelta(seconds=int(total_time)))
        
        fold_annotations = [test_dataset.data[i] for i in range(len(test_dataset))]
        metrics = process_metrics(fold_annotations, predictions)

        metrics_i = "Test metrics: accuracy={:.4f}, f1_fire={:.4f}, f1_nofire={:.4f}, f1_macro={:.4f}".format(
                metrics["accuracy"],
                metrics["f1_fire"],
                metrics["f1_nofire"],
                metrics["f1_macro"],
                metrics["mcc_fire"],
                metrics["mcc_nofire"]
            )
        print(f"###{metrics_i}", flush=True)
        prediction_test_data = 'Number of images: ' + str(len(test_dataset)) + '\n' + metrics_i

    # Save combined
    current_datetime = datetime.now()
    new_output_path = './predictions/' + str(current_datetime.day) + "_" + str(current_datetime.month) + "-" + str(current_datetime.hour) + "_" + str(current_datetime.minute) + '.jsonl'
    write_jsonl(predictions, new_output_path, 'w')
    print("### Combined prediction results saved to:", new_output_path, flush=True)
    write_txt_doc(config['test_files'], prediction_test_data, args.output_path, args.fold_seed if args.cross_val else args.seed, new_output_path)
    print("### Data of the prediction saved to:", args.output_path, flush=True)

    print(f'### {total_time_str}' if args.cross_val else f'### {total_time_str} \n ### {per_fold_times}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_path', type=str, help="path of outputfile")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--cross_val', action='store_true')
    parser.add_argument('--k_folds', default=5, type=int)
    parser.add_argument('--fold_seed', default=123, type=int)
    parser.add_argument('--save_per_fold', action='store_true')

    args = parser.parse_args()

    yaml = YAML(typ='rt')
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    main(args, config)
