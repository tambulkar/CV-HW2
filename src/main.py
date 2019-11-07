from utils import get_frames
from skimage.feature import match_template

def main():
    frames = get_frames('../data/input.mp4')
    first_frame = frames[0]

    cross_corr = match_template(image=first_frame, pad_input=True)

if __name__ == '__main__':
    main()