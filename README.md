# PLFM-RSSR

PLFM-RSSR performs 4× super-resolution on optical remote sensing RGB images. At each decoder scale, it fuses three inputs: the current decoder features, the original encoder features, and the encoder features enhanced by Haar conditioning. 

## Project layout

- `tools/` contains offline degradation, DOTA preprocessing, and latent cache generation.
- `simplesr/` contains the super-resolution framework, including paired-image and cached-latent datasets.
- `plfm_rssr/` contains the LPP, three-way velocity network, VQ-f4 loading, flow-matching training, and Euler inference.
- `configs/` contains the YAML configurations for LPP training, CVN training, and paired evaluation.

## Installation

Run the commands from the project root. The tested setup uses Windows 11, Python 3.11, PyTorch 2.12.0, CUDA 12.6, and an NVIDIA RTX 4090.

```powershell
conda create -n plfm-rssr python=3.11 -y
conda activate plfm-rssr
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Training and inference use a frozen CompVis VQ-f4 autoencoder, downloaded from Hugging Face by default. To use a local copy, set `vq_model` in the training or evaluation YAML, or pass `--vq-model` to `infer.py`. The path can point to `vqvae` or its parent directory. [CompVis VQ-f4](https://huggingface.co/CompVis/ldm-super-resolution-4x-openimages)

## Data preparation

Place paired images in `data/train/{HR,LR}`, `data/val/{HR,LR}`, and `data/test/{HR,LR}`, or set `datasets.<split>.kwargs.dataroot_gt` and `dataroot_lq` in the YAML. The disk dataset matches pairs by filename. Training applies the same crop, rotation, and flip to both images in each pair.

If you have HR images only, use the offline degradation pipeline to generate 4× LR inputs.

```powershell
python tools/generate_lr.py --input data/train/HR --output data/train/LR --scale 4 --device cuda --mode gpu
```

For DOTA, the YOLO-OBB dataset should contain `images/train`, `images/val`, and their corresponding `labels` directories. The following commands generate 256×256 HR patches, 64×64 LR patches, and labeled subsets. Use the resulting `images/train` and `images/val` directories in the training YAML.

```powershell
python -m pip install ultralytics==8.4.90
python tools/dota/prepare_dota_yolo_split_lowres.py --data-root data/DOTA-OBB --save-dir data/dota256 --split trainval --crop-size 256 --gap 0 --lowres-scales 4 --drop-incomplete
python tools/dota/copy_labeled_images.py --source-dir data/dota256 --output-dir data/dota256_labeled
python tools/dota/copy_labeled_images.py --source-dir data/dota256_lr_x4 --output-dir data/dota256_lr_x4_labeled
```

For paired training with LMDB, pack the HR and LR PNG directories separately and set `io_backend_opt: {type: lmdb}` in the paired dataset configuration. Latent cache generation uses disk-based image folders.

```powershell
python -m simplesr.data.lmdb_util --input data/train/HR --output data/train/HR.lmdb
python -m simplesr.data.lmdb_util --input data/train/LR --output data/train/LR.lmdb
```

## Training

Train the latent prior predictor (LPP) first, then train the conditional velocity network (CVN) with the LPP frozen. Cached training requires complete 256×256 HR and 64×64 LR RGB patches. Generate the cache using the same YAML, dataset paths, and VQ-f4 model as CVN training.

```powershell
python train.py -opt configs/train_lpp.yml
python tools/precompute_frozen_latents.py --config configs/train_cvn.yml --output-root latent_cache
python train.py -opt configs/train_cvn.yml path.lpp_weights=weights/lpp.pth
```

Replace `weights/lpp.pth` with the path to the LPP EMA checkpoint, such as `weights/net_g_ema_100000.pth` from the first stage. `train_cvn.yml` uses the aggregate latent cache by default and also supports the per-image format. Use `train_cvn_online.yml` to encode latents during training.

Each stage runs for 100,000 optimizer updates by default. The LPP uses a microbatch of 4 and 8 gradient accumulation steps. The CVN uses a microbatch of 16 and 2 accumulation steps. Override these settings from the command line with `train.total_iter=...`, `datasets.train.batch_size_per_gpu=...`, or `train.accumulation_steps=...`. Logs, weights, and training state are saved under `output/<experiment_name>/train_<timestamp>/`.

## Evaluation and inference

Use a complete three-way checkpoint from the second training stage, such as `weights/net_g_ema_100000.pth`. Obtain or train this checkpoint separately. Paired evaluation saves super-resolved images and reports PSNR, SSIM, LPIPS, and DISTS. The first perceptual metric evaluation may download the metric network weights.

```powershell
python test.py -opt configs/test.yml path.pretrain_network_g=weights/plfm_rssr.pth
python infer.py --input data/example.png --output results/example_x4.png --weights weights/plfm_rssr.pth --device cuda
python infer.py --input data/test/LR --output results --weights weights/plfm_rssr.pth --device cuda --steps 10
```
