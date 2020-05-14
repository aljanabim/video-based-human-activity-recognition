import os


def concat():
    cmd = "ffmpeg -f concat -safe 0 -i info.txt -c copy 2_of_each.avi"
    return cmd


def cut(start, end, in_name, out_name):
    cmd = f"ffmpeg -i {in_name}.avi -ss {start} -to {end} -c:v libx264 -c:a aac {out_name}.avi"
    os.system(cmd)
    return cmd


if __name__ == '__main__':
    cut("00:04:55", "00:05:56", "pred_output_6", "final_pres")
