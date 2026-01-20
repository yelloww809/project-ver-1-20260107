# base
| num         | PROCESS   | MODEL   | EPOCHS | PATIENCE | COS_LR | AUGMENTATION |
| ----------- | --------- | ------- | ------ | -------- | ------ | ------------ |
| v9_base_1_1 | v9_base_1 | yolo11n | 100    | 30       | False  | default      |
| v9_base_2_1 | jpg       |         |        |          |        |              |
| v9_base_3   | jpg       |         |        |          |        |              |
|             |           |         |        |          |        |              |

# large
| num        | SAVE_IMAGE_FORMAT | STFT_MODE | (MODE 1) NPERSEG | (MODE 3) FREQ_RES_KHZ | USE_DB_SCALE | NORM_TYPE | (GLOBAL) MIN_DB | (GLOBAL)MAX_DB |
| ---------- | ----------------- | --------- | ---------------- | --------------------- | ------------ | --------- | --------------- | -------------- |
| v9_large_1 |                   |           |                  |                       |              |           |                 |                |

# small
> 此时强制使用 STFT_MODE = 3
| num        | SAVE_IMAGE_FORMAT | (MODE 3) FREQ_RES_KHZ | USE_DB_SCALE | NORM_TYPE | (GLOBAL) MIN_DB | (GLOBAL) MAX_DB |
| ---------- | ----------------- | --------------------- | ------------ | --------- | --------------- | --------------- |
| v9_small_1 |                   |                       |              |           |                 |                 |