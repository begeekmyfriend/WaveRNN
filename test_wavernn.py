from apex import amp
from utils.dataset import get_vocoder_datasets
from utils.dsp import *
from models.fatchord_version import WaveRNN
from utils.paths import Paths
from utils.display import simple_table
import numpy as np
import os
import torch
import argparse


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description='Generate WaveRNN Samples')
    parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation')
    parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slow Unbatched Generation')
    parser.add_argument('--samples', '-s', type=int, help='[int] number of utterances to generate')
    parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
    parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
    parser.add_argument('--dir', '-d', type=str, default='.', help='[string/path] for testing a wav outside dataset')
    parser.add_argument('--weights', '-w', type=str, help='[string/path] checkpoint file to load weights from')
    parser.add_argument('--gta', '-g', dest='use_gta', action='store_true', help='Generate from GTA testset')

    parser.set_defaults(batched=hp.voc_gen_batched)
    parser.set_defaults(samples=hp.voc_gen_at_checkpoint)
    parser.set_defaults(target=hp.voc_target)
    parser.set_defaults(overlap=hp.voc_overlap)
    parser.set_defaults(file=None)
    parser.set_defaults(weights=None)
    parser.set_defaults(gta=False)

    args = parser.parse_args()

    batched = args.batched
    samples = args.samples
    target = args.target
    overlap = args.overlap
    gta = args.gta

    model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                    fc_dims=hp.voc_fc_dims,
                    bits=hp.bits,
                    pad=hp.voc_pad,
                    upsample_factors=hp.voc_upsample_factors,
                    feat_dims=hp.num_mels,
                    compute_dims=hp.voc_compute_dims,
                    res_out_dims=hp.voc_res_out_dims,
                    res_blocks=hp.voc_res_blocks,
                    hop_length=hp.hop_length,
                    sample_rate=hp.sample_rate,
                    pad_val=hp.voc_pad_val,
                    mode=hp.voc_mode).cuda()

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    restore_path = args.weights if args.weights else paths.voc_latest_weights

    model.restore(restore_path)
    model.eval()
    if hp.amp:
        model, _ = amp.initialize(model, [], opt_level='O3')

    simple_table([('Generation Mode', 'Batched' if batched else 'Unbatched'),
                  ('Target Samples', target if batched else 'N/A'),
                  ('Overlap Samples', overlap if batched else 'N/A')])

    k = model.get_step() // 1000

    for file_name in os.listdir(args.dir):
        if file_name.endswith('.npy'):
            mel = np.load(os.path.join(args.dir, file_name))
            mel = torch.tensor(mel).unsqueeze(0)

            batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
            save_str = f'{file_name}__{k}k_steps_{batch_str}.wav'

            model.generate(mel, save_str, batched, target, overlap, hp.mu_law)
