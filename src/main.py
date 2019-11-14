from utils import get_frames
from skimage.feature import match_template
import numpy as np
import matplotlib.pyplot as plt
from cv2 import imwrite, warpAffine
from os import path

def main():
    frames = get_frames('../data/rohan_input.mov')
    first_frame = frames[0]
    imwrite('../output/first_frame.png', first_frame)
    template = first_frame[670:870, 1075:1275] #first_frame[800:1350,555:1030]
    imwrite('../output/template.png', template)
    cross_corr = match_template(image=first_frame, template=template, pad_input=True)
    plt.imshow(cross_corr, cmap="gray")
    plt.savefig('../output/corr_matrix.png')
    plt.clf()

    if not path.exists('../data/x_shifts.npy') or not path.exists('../data/y_shifts.npy'):
        x_shifts = []
        y_shifts = []
        for i, frame in enumerate(frames):
            cross_corr = match_template(image=frame, template=template, pad_input=True)
            shift = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
            x, y = shift[::-1]
            x_shifts.append(x)
            y_shifts.append(y)
            print('Found shift for {num} of {length} frames'.format(num=i+1, length=len(frames)))
    else:
        x_shifts = np.load('../data/x_shifts.npy')
        y_shifts = np.load('../data/y_shifts.npy')

    plt.plot(x_shifts, y_shifts)
    plt.xlabel('Y Pixel Shift')
    plt.ylabel('X Pixel Shift')
    plt.savefig('../output/pixel_shift.png')
    plt.clf()

    # frames = get_frames('../data/input.mov', grayscale=False)
    shifted_frames = []
    for i, frame in enumerate(frames):
        frame = np.array(frame, dtype='uint8')
        shifted_frame = frame.copy()
        x_shift = x_shifts[i]
        y_shift = y_shifts[i]
        M = np.array([[1, 0, x_shift],
                    [0, 1, y_shift]], dtype='float32')
        warpAffine(src=frame, dst=shifted_frame, M=M, dsize=frame.shape)
        shifted_frame = np.array(shifted_frame, dtype='int32')
        shifted_frames.append(shifted_frame)
        print('Shifted {num} of {length} frames'.format(num=i + 1, length=len(frames)))
    res = shifted_frames[0]
    for i in range(1, len(shifted_frames)):
        res += shifted_frames[i]
    res = np.divide(res, len(shifted_frames))
    imwrite('../output/output.png', res)

if __name__ == '__main__':
    main()