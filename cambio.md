# TranformData
id, type, params
0, none, "{}"
1, sharifzadeh, "{'eps':1e-8}"

# ResizeGtConfig

id, type, params
0, original, "{}"
1, resize, "{'nz_original':2048, 'mode':'reflect'}"

# SamplesOrganization

id, seed, ratio, order, select_mode, n_train, n_val, query, train_idxs, val_idxs
0, 0.9, CWH, "select_idxs", "(RF == 1) and (nc == 128) and (name.str.slice(0, 3) != 'JHU')", "0:12,14:19,22:51,53:57,59:70,72:86,89:90,92:98,100:101,103:105,107:120,122:129,131:159,161:165,167:188,190,192:213,215:240,242,244:251,253:263,265:269,271:272,274:275,277:307,309:312,314,316:329,331:338,340:342,344,346:365,367:371,373:384,386,388:412,414:426,428:434,436:442,444:453,455:457,460:465,467:473,476:483,485:490,492:503,506:509,511:529,531:534,536:543,545:549", "13,20:21,52,58,71,87:88,91,99,102,106,121,130,160,166,189,191,214,241,243,252,264,270,273,276,308,313,315,330,339,343,345,366,372,385,387,413,427,435,443,454,458:459,466,474:475,484,491,504:505,510,530,535,544"
1, 42, 0.9, CWH, "random_split", -1, -1, "(RF == 1) and (nc == 128) and (name.str.slice(0, 3) != 'JHU') and (source == 'CUBDL')", "-1", "-1"

# DataSize
id, nz, nx, ns
0, 2048, 256, 2300
1, 2048, 256, 2800
2, 1024, 256, 2300

# DataType
id, type, params
0, "RF", "{}"
1, "RF Analitic", "{}"
2, "I/Q", "{}"

# WebDatasetBeamformer
id, gt_source, data_type_id, data_size_id, samples_organization_id, transform_data_id, resize_gt_id
0, "DAS_mean", 0, 0, 1, 1, 0
1, "DAS_mean", 1, 0, 1, 1, 0
2, "DAS_mean", 2, 0, 1, 1, 0
3, "DAS_mean", 0, 2, 1, 1, 0
4, "DAS_mean", 1, 2, 1, 1, 0
5, "DAS_mean", 2, 2, 1, 1, 0
