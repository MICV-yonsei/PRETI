import subprocess
import os

def main():
    command = (
        # "python -m torch.distributed.launch "
        "torchrun --nproc_per_node=8 --master_port=12345 train_no_meta.py "
        
        "--architecture vitb "

        "--batch_size 512 "
        "--epochs 300 "

        "--output_dir /home/leeyeonkyung/PRETI/EVALUATION_RESULT/PRETI_no_meta "
        "--log_dir /home/leeyeonkyung/PRETI/EVALUATION_RESULT/PRETI_no_meta "
        #=====================
        "--random_area_min_1 0.3 "
        "--random_area_max_1 1.0 "
        "--random_area_min_2 0.2 "
        "--random_area_max_2 0.8 "
        #
        "--random_aspect_ratio_min_1 0.9 "
        "--random_aspect_ratio_max_1 1.1 "
        "--random_aspect_ratio_min_2 0.75 "
        "--random_aspect_ratio_max_2 1.25 "
       # 
        "--horizontal_flip_p_1 0.1 "
        "--horizontal_flip_p_2 0.5 "
        #=====================
    )
    os.system(command)


if __name__ == "__main__":
    main()
