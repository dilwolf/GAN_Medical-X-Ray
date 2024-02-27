## X-Ray Medical Image Transformation using StarGAN v2

This repository contains the implementation of X-Ray medical image transformation using StarGAN v2. StarGAN v2 is employed for its advanced capabilities in image translation tasks.

### Disclaimer

This repository is for educational purposes only. The utilization of the StarGAN v2 model is solely for academic exploration and understanding of image transformation techniques.

### Usage

1. **Resize Images**: Use the following command to resize images before training:
   ```bash
   python resize.py
   ```

2. **Train the Model**: Train the model with the following command:
   ```bash
   python main.py --mode train --num_domains 3 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 --train_img_dir data/x-raymed/train --val_img_dir data/x-raymed/val --checkpoint_dir expr/checkpoints/x-raymed --sample_every 1000 --save_every 5000
   ```

3. **Generate Reference Images**: Generate reference images using the trained model:
   ```bash
   python main.py --mode sample --num_domains 3 --resume_iter 5000 --w_hpf 0 --checkpoint_dir expr/checkpoints/x-raymed --result_dir expr/results/x-raymed --src_dir assets/representative/x-raymed/src --ref_dir assets/representative/x-raymed/ref
   ```

4. **Evaluate and Generate Single Images**: Evaluate and generate single images using the trained model:
   ```bash
   python main.py --mode eval --num_domains 3 --w_hpf 0 --resume_iter 5000 --train_img_dir data/x-raymed/train --val_img_dir data/x-raymed/val --checkpoint_dir expr/checkpoints/x-raymed --eval_dir expr/eval/x-raymed 
   ```

For more detailed instructions and options, please refer to the documentation.