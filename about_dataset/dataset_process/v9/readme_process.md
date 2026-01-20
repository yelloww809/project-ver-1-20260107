# base
| num       | SAVE_IMAGE_FORMAT | STFT_MODE | (MODE 1) NPERSEG | (MODE 3) FREQ_RES_KHZ | USE_DB_SCALE | NORM_TYPE | (GLOBAL) MIN_DB | (GLOBAL)MAX_DB |
| --------- | ----------------- | --------- | ---------------- | --------------------- | ------------ | --------- | --------------- | -------------- |
| v9_base_1 | jpg               | 1         | 1024             | /                     | False        | SAMPLE    | /               | /              |
| v9_base_2 | jpg               | 2         | /                | /                     | False        | SAMPLE    | /               | /              |
| v9_base_3 | jpg               | 3         | /                | 20                    | False        | SAMPLE    | /               | /              |
| v9_base_4 | jpg               | 2         | /                | /                     | True         | SAMPLE    | /               | /              |
| v9_base_5 | jpg               | 2         | /                | /                     | False        | GLOBAL    | -140            | 30             |
| v9_base_6 | jpg               | 2         | /                | /                     | True         | GLOABL    | -140            | 30             |
| v9_base_7 | jpg               | 3         | /                | 20                    | True         | SAMPLE    | /               | /              |
| v9_base_8 | jpg               | 3         | /                | 20                    | False        | GLOBAL    | -140            | 30             |
| v9_base_9 | jpg               | 3         | /                | 20                    | True         | GLOBAL    | -140            | 30             |
|           |                   |           |                  |                       |              |           |                 |                |

# large
| num        | SAVE_IMAGE_FORMAT | STFT_MODE | (MODE 1) NPERSEG | (MODE 3) FREQ_RES_KHZ | USE_DB_SCALE | NORM_TYPE | (GLOBAL) MIN_DB | (GLOBAL)MAX_DB |
| ---------- | ----------------- | --------- | ---------------- | --------------------- | ------------ | --------- | --------------- | -------------- |
| v9_large_2 | jpg               | 2         | /                | /                     | False        | SAMPLE    | /               | /              |
| v9_large_3 | jpg               | 3         | /                | 20                    | False        | SAMPLE    | /               | /              |
| v9_large_4 | jpg               | 2         | /                | /                     | True         | SAMPLE    | /               | /              |
| v9_large_7 | jpg               | 3         | /                | 20                    | True         | SAMPLE    | /               | /              |
|            |                   |           |                  |                       |              |           |                 |                |

# small
> 此时强制使用 STFT_MODE = 3
| num        | SAVE_IMAGE_FORMAT | (MODE 3) FREQ_RES_KHZ | USE_DB_SCALE | NORM_TYPE | (GLOBAL) MIN_DB | (GLOBAL) MAX_DB |
| ---------- | ----------------- | --------------------- | ------------ | --------- | --------------- | --------------- |
| v9_small_1 | jpg               | 5                     | False        | SAMPLE    | /               | /               |
| v9_small_2 | jpg               | 5                     | True         | SAMPLE    | /               | /               |
| v9_small_3 | jpg               | 5                     | False        | GLOBAL    | -140            | 30              |
| v9_small_4 | jpg               | 5                     | True         | GLOBAL    | -140            | 30              |
|            |                   |                       |              |           |                 |                 |