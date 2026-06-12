---
marp: true
paginate: true
theme: beam
header:
footer:
---
<!-- _class: title --->

# Presentación 11: Corrección de Beamformers y nuevo entrenamiento solo para CPWC

Sebastián Gutiérrez Milla

---
<style scoped> h1 { font-size: 1.5rem; } </style>
# Compressing Beamforming (Antes Sparse Regularization)


![w:1250 h:400](all_gts.png) 

---
# F-DMAS Corrección

| F-DMAS Paper | F-DMAS propio |
| :---: | :---: |
| ![w:400 h:400](fdmas.png) | ![w:410 h:410](fdmas_contrast_speckle_expe_dataset_rf_gts.png) |

---
# CV Corrección

| CV Paper | CV propio |
| :---: | :---: |
| ![w:400 h:400](cf.png) | ![w:410 h:410](cf_contrast_speckle_expe_dataset_rf_gts.png) |

---
# iMAP Corrección

| iMAP Paper | iMAP propio |
| :---: | :---: |
| ![w:400 h:400](imap.png) | ![w:410 h:410](imap_contrast_speckle_expe_dataset_rf_gts.png) |

---
# MV Corrección

| MV Paper | MV propio |
| :---: | :---: |
| ![w:400 h:400](mv.png) | ![w:410 h:410](mv_carotid_cross_expe_dataset_rf_gts.png) |

---
<style scoped> h1 { font-size: 1.3rem; } </style>

# contrast_speckle_expe_dataset_rf: Ground Truths Central Angle

![w:1250 h:530](contrast_speckle_expe_dataset_rf_Central_gts.png)

---
<style scoped> h1 { font-size: 1.3rem; } </style>

# contrast_speckle_expe_dataset_rf: Ground Truths CPWC

![w:1250 h:530](contrast_speckle_expe_dataset_rf_CPWC_gts.png)

---
<style scoped> h1 { font-size: 1.3rem; } </style>

# contrast_speckle_simu_dataset_rf: Ground Truths Central Angle

![w:1250 h:530](contrast_speckle_simu_dataset_rf_Central_gts.png)

---
<style scoped> h1 { font-size: 1.3rem; } </style>

# contrast_speckle_simu_dataset_rf: Ground Truths CPWC

![w:1250 h:530](contrast_speckle_simu_dataset_rf_CPWC_gts.png)

---
<style scoped> h1 { font-size: 1.3rem; } </style>

# contrast_speckle_expe_dataset_rf: Models showcase

![w:1100 h:570](contrast_speckle_expe_dataset_rf_CPWC_models.png)

---
<style scoped> h1 { font-size: 1.3rem; } </style>

# contrast_speckle_expe_dataset_rf: Métricas Contraste

| name                 |      cnr |     gcnr |
|:---------------------|---------:|---------:|
| BINN_OG-3 DAS-CPWC   | 0.084097 | 0.214575 |
| BINN_OG-3 MV-CPWC    | 0.179034 | 0.234413 |
| BINN_OG-3 FDMAS-CPWC | 0.142311 | 0.202834 |
| BINN_OG-3 CF-CPWC    | 0.148376 | 0.260729 |
| BINN_OG-3 IMAP-CPWC  | 0.082930 | 0.217004 |

---
<style scoped> h1 { font-size: 1.3rem; } </style>

# contrast_speckle_expe_dataset_rf: SSIM

| name                 |     ssim |
|:---------------------|---------:|
| BINN_OG-3 DAS-CPWC   | 0.173811 |
| BINN_OG-3 MV-CPWC    | 0.440758 |
| BINN_OG-3 FDMAS-CPWC | 0.280202 |
| BINN_OG-3 CF-CPWC    | 0.322773 |
| BINN_OG-3 IMAP-CPWC  | 0.181312 |

---
<style scoped> h1 { font-size: 1.3rem; } </style>

# contrast_speckle_simu_dataset_rf: Models showcase

![w:1100 h:570](contrast_speckle_simu_dataset_rf_CPWC_models.png)

---
<style scoped> h1 { font-size: 1.3rem; } </style>

# contrast_speckle_simu_dataset_rf: Métricas Contraste

| name                 |      cnr |     gcnr |
|:---------------------|---------:|---------:|
| BINN_OG-3 DAS-CPWC   | 0.190571 | 0.189548 |
| BINN_OG-3 MV-CPWC    | 0.134399 | 0.174312 |
| BINN_OG-3 FDMAS-CPWC | 0.040206 | 0.118938 |
| BINN_OG-3 CF-CPWC    | 0.060199 | 0.188729 |
| BINN_OG-3 IMAP-CPWC  | 0.164318 | 0.172018 |

---
<style scoped> h1 { font-size: 1.3rem; } </style>

# contrast_speckle_simu_dataset_rf: SSIM

| name                 |     ssim |
|:---------------------|---------:|
| BINN_OG-3 DAS-CPWC   | 0.202192 |
| BINN_OG-3 MV-CPWC    | 0.358028 |
| BINN_OG-3 FDMAS-CPWC | 0.306531 |
| BINN_OG-3 CF-CPWC    | 0.358600 |
| BINN_OG-3 IMAP-CPWC  | 0.212043 |


