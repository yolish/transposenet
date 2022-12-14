import os
dataset = "cambridge"#"7scenes"#


if dataset == "7scenes":

    checkpoint_path = "out/run_03_12_22_19_31_final.pth"
    log_name = checkpoint_path.replace("out/", "")
    out_filename = "{}_{}_logs.txt".format(dataset, log_name)
    scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    if not os.path.exists(out_filename):
        for s in scenes:
            cmd = "python main.py transposenet test ./models/backbones/efficient-net-b0.pth /media/yoli/WDC-2.0-TB-Hard-/7Scenes " \
                  "./datasets/7Scenes/abs_7scenes_pose.csv_{}_test.csv --checkpoint_path {} >> {}".format(s,
                                                                                                                           checkpoint_path,
                                                                                                                           out_filename)
            os.system(cmd)

else:
    checkpoint_path = "out/run_04_12_22_10_20_checkpoint-550.pth"#"out/run_05_12_22_10_04_checkpoint-100.pth"
    log_name = checkpoint_path.replace("out/", "")
    out_filename = "{}_{}_logs.txt".format(dataset, log_name)
    scenes = ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']

    if not os.path.exists(out_filename):
        for s in scenes:
            cmd = 'python main.py transposenet test ./models/backbones/efficient-net-b0.pth /media/yoli/WDC-2.0-TB-Hard-/CambridgeLandmarks ' \
                  './datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{}_test.csv --checkpoint_path {} >> {}'.format(s,
                                                                                                                      checkpoint_path,
                                                                                                                      out_filename)
            os.system(cmd)

f = open(out_filename)
lines = f.readlines()
i = 0

for l in lines:
    if "Median pose error: " in l:
        s = scenes[i]
        i += 1
        details = l.rstrip().split("Median pose error: ")[1]
        print("{} - {}".format(s, details))
    if "Var pose error: " in l:
        details = l.rstrip().split("Var pose errorr: ")[1]
        print("{} -std  {}".format(s, details))
